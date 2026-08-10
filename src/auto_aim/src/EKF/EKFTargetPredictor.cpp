#include "EKF/EKFTargetPredictor.h"

#include <optional>

namespace {
constexpr double kMillimetersPerMeter = 1000.0;
}

EKFTargetPredictor::EKFTargetPredictor(
    const EKFTargetObservation& initial_observation,
    double initial_radius_mm,
    std::shared_ptr<YAML::Node> config_file_ptr)
    : tracker_(std::make_unique<rm_ekf::RobustArmorTracker>()),
      last_update_time_(initial_observation.t) {
    rm_ekf::RobustTrackerConfig config;
    if (config_file_ptr) {
        config = rm_ekf::RobustTrackerConfig::fromYaml(
            *config_file_ptr, initial_radius_mm / kMillimetersPerMeter);
    } else {
        config.initial_filter.r1 = initial_radius_mm / kMillimetersPerMeter;
        config.initial_filter.r2 = initial_radius_mm / kMillimetersPerMeter;
        config.initial_filter.h = 0.0;
    }
    tracker_->configure(config);

    last_result_ = tracker_->process(toMeters(initial_observation), 0.0, -1);
    update_frames_ = 1;
    if (tracker_->currentArmorId() >= 0) {
        debug_flip_flag_ = (tracker_->currentArmorId() % 2) + 1;
    }
}

void EKFTargetPredictor::update(const EKFTargetObservation& observation) {
    const double dt = observation.t - last_update_time_;
    last_result_ = tracker_->process(toMeters(observation), dt, -1);
    if (last_result_.updated || last_result_.initialized_this_frame) {
        ++update_frames_;
    }
    last_update_time_ = observation.t;
    if (tracker_->currentArmorId() >= 0) {
        debug_flip_flag_ = (tracker_->currentArmorId() % 2) + 1;
    }
}

void EKFTargetPredictor::missUpdate(double update_time) {
    const double dt = update_time - last_update_time_;
    last_result_ = tracker_->process(std::nullopt, dt, -1);
    last_update_time_ = update_time;
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

rm_ekf::ArmorObservation EKFTargetPredictor::toMeters(
    const EKFTargetObservation& observation) {
    rm_ekf::ArmorObservation result;
    result.x = observation.x / kMillimetersPerMeter;
    result.y = observation.y / kMillimetersPerMeter;
    result.z = observation.z / kMillimetersPerMeter;
    result.yaw = observation.yaw;
    return result;
}
