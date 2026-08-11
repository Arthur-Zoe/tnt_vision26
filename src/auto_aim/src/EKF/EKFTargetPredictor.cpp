#include "EKF/EKFTargetPredictor.h"

#include <cmath>
#include <iostream>
#include <optional>

namespace {
constexpr double kMillimetersPerMeter = 1000.0;
}

EKFTargetPredictor::EKFTargetPredictor(
    const EKFTargetObservation& initial_observation,
    double initial_radius_mm,
    std::shared_ptr<YAML::Node> config_file_ptr) {
    if (config_file_ptr) {
        config_ = rm_ekf::RobustTrackerConfig::fromYaml(
            *config_file_ptr, initial_radius_mm / kMillimetersPerMeter);
        const YAML::Node reset_time_node = (*config_file_ptr)["reset_predictor_time"];
        if (reset_time_node) {
            const double reset_time_ms = reset_time_node.as<double>();
            if (std::isfinite(reset_time_ms) && reset_time_ms > 0.0) {
                max_tracking_gap_s_ = reset_time_ms / 1000.0;
            }
        }
    } else {
        config_.initial_filter.r1 = initial_radius_mm / kMillimetersPerMeter;
        config_.initial_filter.r2 = initial_radius_mm / kMillimetersPerMeter;
        config_.initial_filter.h = 0.0;
    }

    if (std::isfinite(initial_observation.t)) {
        initializeFromObservation(initial_observation);
    } else {
        resetTracker();
        last_dt_s_ = initial_observation.t;
        warnTimeIssue("non-finite timestamp", initial_observation.t,
                      initial_observation.t);
    }
}

void EKFTargetPredictor::update(const EKFTargetObservation& observation) {
    time_discontinuity_ = false;
    if (!std::isfinite(observation.t)) {
        last_dt_s_ = observation.t;
        warnTimeIssue("non-finite timestamp", observation.t, observation.t);
        return;
    }

    if (!has_update_time_) {
        initializeFromObservation(observation);
        timestamp_warning_active_ = false;
        return;
    }

    const double dt = observation.t - last_update_time_;
    last_dt_s_ = dt;
    if (!std::isfinite(dt)) {
        warnTimeIssue("non-finite dt", observation.t, dt);
        return;
    }
    if (dt <= 0.0) {
        warnTimeIssue("duplicate/out-of-order timestamp", observation.t, dt);
        return;
    }
    if (dt > max_tracking_gap_s_) {
        warnTimeIssue("tracking time discontinuity", observation.t, dt);
        time_discontinuity_ = true;
        initializeFromObservation(observation);
        return;
    }

    last_result_ = tracker_->process(toMeters(observation), dt, -1);
    last_update_time_ = observation.t;
    timestamp_warning_active_ = false;
    if (last_result_.updated || last_result_.initialized_this_frame) {
        ++update_frames_;
    }
    if (tracker_->currentArmorId() >= 0) {
        debug_flip_flag_ = (tracker_->currentArmorId() % 2) + 1;
    }
}

void EKFTargetPredictor::missUpdate(double update_time) {
    time_discontinuity_ = false;
    if (!std::isfinite(update_time)) {
        last_dt_s_ = update_time;
        warnTimeIssue("non-finite timestamp", update_time, update_time);
        return;
    }

    if (!has_update_time_) {
        last_update_time_ = update_time;
        last_dt_s_ = 0.0;
        has_update_time_ = true;
        timestamp_warning_active_ = false;
        return;
    }

    const double dt = update_time - last_update_time_;
    last_dt_s_ = dt;
    if (!std::isfinite(dt)) {
        warnTimeIssue("non-finite dt", update_time, dt);
        return;
    }
    if (dt <= 0.0) {
        warnTimeIssue("duplicate/out-of-order timestamp", update_time, dt);
        return;
    }
    if (dt > max_tracking_gap_s_) {
        warnTimeIssue("tracking time discontinuity", update_time, dt);
        resetTracker();
        last_update_time_ = update_time;
        has_update_time_ = true;
        time_discontinuity_ = true;
        return;
    }

    last_result_ = tracker_->process(std::nullopt, dt, -1);
    last_update_time_ = update_time;
    timestamp_warning_active_ = false;
    if (tracker_->currentArmorId() >= 0) {
        debug_flip_flag_ = (tracker_->currentArmorId() % 2) + 1;
    }
}

EKFTargetPrediction EKFTargetPredictor::predict(double predict_time) const {
    EKFTargetPrediction result;
    if (!tracker_->hasState()) {
        return result;
    }

    const rm_ekf::ArmorState state = tracker_->state();
    result.center_x = (state.x + predict_time * state.vx) * kMillimetersPerMeter;
    result.center_y = (state.y + predict_time * state.vy) * kMillimetersPerMeter;
    result.center_z = (state.z + predict_time * state.vz) * kMillimetersPerMeter;
    result.alternate_z =
        (state.z + state.h + predict_time * state.vz) * kMillimetersPerMeter;
    result.r1 = state.r1 * kMillimetersPerMeter;
    result.r2 = state.r2 * kMillimetersPerMeter;
    result.h = state.h * kMillimetersPerMeter;
    result.yaw = rm_ekf::wrapAngle(state.yaw + state.w * predict_time);
    result.w = state.w;
    result.rotation_direction = state.w >= 0.0 ? 1 : -1;

    const auto armors = tracker_->predictArmors(predict_time);
    result.armors.reserve(armors.size());
    for (const auto& armor : armors) {
        const double radius = armor.id % 2 == 0 ? state.r1 : state.r2;
        result.armors.push_back(EKFPredictedArmor{
            armor.x * kMillimetersPerMeter,
            armor.y * kMillimetersPerMeter,
            armor.z * kMillimetersPerMeter,
            radius * kMillimetersPerMeter,
            armor.yaw,
        });
    }
    return result;
}

EKFTargetState EKFTargetPredictor::state() const {
    EKFTargetState result;
    if (!tracker_->hasState()) {
        return result;
    }

    const rm_ekf::ArmorState state = tracker_->state();
    result.center_x = state.x * kMillimetersPerMeter;
    result.center_y = state.y * kMillimetersPerMeter;
    result.center_z = state.z * kMillimetersPerMeter;
    result.center_vx = state.vx * kMillimetersPerMeter;
    result.center_vy = state.vy * kMillimetersPerMeter;
    result.center_vz = state.vz * kMillimetersPerMeter;
    result.r1 = state.r1 * kMillimetersPerMeter;
    result.r2 = state.r2 * kMillimetersPerMeter;
    result.h = state.h * kMillimetersPerMeter;
    result.yaw = state.yaw;
    result.w = state.w;
    result.update_frames = update_frames_;
    return result;
}

EKFTargetDebugState EKFTargetPredictor::debugState() const {
    EKFTargetDebugState debug;
    debug.dt_s = last_dt_s_;
    debug.time_discontinuity = time_discontinuity_;
    debug.tracker_state = rm_ekf::trackerStateName(last_result_.state);
    debug.matched_id = last_result_.matched_id;
    debug.measurement_valid = last_result_.measurement_valid;
    debug.updated = last_result_.updated;
    debug.lost_frames = last_result_.lost_frames;
    debug.nis = last_result_.nis;
    debug.position_error_m = last_result_.position_error;
    debug.yaw_error_deg = last_result_.yaw_error >= 0.0
                              ? rm_ekf::rad2deg(last_result_.yaw_error)
                              : -1.0;
    debug.phase_observer_valid = last_result_.phase_observer_valid;
    debug.phase_w_instant = last_result_.phase_w_instant;
    debug.phase_w_filtered = last_result_.phase_w_filtered;
    debug.direction_reversal = last_result_.direction_reversal;
    debug.armor_switched = last_result_.armor_switched;
    debug.recovered = last_result_.recovered;
    debug.phase_w_applied = last_result_.phase_w_applied;
    return debug;
}

bool EKFTargetPredictor::ready() const {
    return tracker_->ready();
}

bool EKFTargetPredictor::hasState() const {
    return tracker_->hasState();
}

void EKFTargetPredictor::warnTimeIssue(const char* reason,
                                       double update_time,
                                       double dt) {
    if (!timestamp_warning_active_) {
        std::cerr << "[EKFTargetPredictor] warning: " << reason
                  << "; t=" << update_time << " dt=" << dt
                  << " s, max_tracking_gap=" << max_tracking_gap_s_ << " s"
                  << std::endl;
        timestamp_warning_active_ = true;
    }
}

void EKFTargetPredictor::resetTracker() {
    tracker_ = std::make_unique<rm_ekf::RobustArmorTracker>();
    tracker_->configure(config_);
    last_result_ = rm_ekf::TrackerResult{};
    update_frames_ = 0;
    debug_flip_flag_ = 1;
}

void EKFTargetPredictor::initializeFromObservation(
    const EKFTargetObservation& observation) {
    resetTracker();
    last_result_ = tracker_->process(toMeters(observation), 0.0, -1);
    last_update_time_ = observation.t;
    has_update_time_ = true;
    update_frames_ = 1;
    if (tracker_->currentArmorId() >= 0) {
        debug_flip_flag_ = (tracker_->currentArmorId() % 2) + 1;
    }
}

rm_ekf::ArmorObservation EKFTargetPredictor::toMeters(
    const EKFTargetObservation& observation) {
    rm_ekf::ArmorObservation result;
    result.x = observation.x / kMillimetersPerMeter;
    result.y = observation.y / kMillimetersPerMeter;
    result.z = observation.z / kMillimetersPerMeter;
    result.yaw = observation.yaw;
    return result;
}
