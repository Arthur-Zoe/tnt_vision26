#include "EKF/SuperPowerPredictor.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
constexpr double kMillimetersPerMeter = 1000.0;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kHalfPi = kPi / 2.0;
}

SuperPowerPredictor::SuperPowerPredictor(
    const EKFTargetObservation& initial_observation,
    double initial_radius_mm,
    std::shared_ptr<YAML::Node> config_file_ptr) {
    // Do not inherit any old previous estimator tuning. These defaults are
    // SuperPower standard3 + the normal four-armor Tracker::set_target branch.
    config_.min_detect_count = 5;
    config_.max_temp_lost_count = 15;
    config_.max_dt_s = 0.1;
    config_.initial_radius_m = 0.2;
    config_.armor_num = 4;

    // The YAML block is deliberately a transcription of those SP constants.
    // Reading it here keeps one visible source of truth without consulting the
    // legacy previous estimator config block.
    if (config_file_ptr) {
        const YAML::Node sp = (*config_file_ptr)["superpower_ekf"];
        if (sp) {
            if (sp["min_detect_count"])
                config_.min_detect_count = sp["min_detect_count"].as<int>();
            if (sp["max_temp_lost_count"])
                config_.max_temp_lost_count = sp["max_temp_lost_count"].as<int>();
            if (sp["max_dt_s"])
                config_.max_dt_s = sp["max_dt_s"].as<double>();
            if (sp["initial_radius_m"])
                config_.initial_radius_m = sp["initial_radius_m"].as<double>();
            if (sp["armor_num"])
                config_.armor_num = sp["armor_num"].as<int>();
        }
    }

    // Keep the old constructor ABI. SP's normal branch initializes at 0.2 m,
    // so the previous caller-provided RMM radius must not alter the baseline.
    (void)initial_radius_mm;

    resetTracker();
    if (std::isfinite(initial_observation.t)) {
        initializeFromObservation(initial_observation);
    } else {
        last_dt_s_ = initial_observation.t;
        warnTimeIssue("non-finite timestamp", initial_observation.t,
                      initial_observation.t);
    }
}

void SuperPowerPredictor::update(const EKFTargetObservation& observation) {
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

    // Do not pre-reset at the old RMM reset_predictor_time.  The SP tracker
    // itself applies its original 0.1 s large-dt reset behavior.
    if (dt > config_.max_dt_s) {
        time_discontinuity_ = true;
    }

    last_observation_ = toSuperPower(observation);
    last_result_ = tracker_->process(last_observation_, dt);
    last_update_time_ = observation.t;
    timestamp_warning_active_ = false;

    if (last_result_.initialized_this_frame) {
        update_frames_ = 1;
    } else if (last_result_.updated) {
        ++update_frames_;
    }
    if (last_result_.matched_id >= 0) {
        debug_flip_flag_ = (last_result_.matched_id % 2) + 1;
    }
}

void SuperPowerPredictor::missUpdate(double update_time) {
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

    if (dt > config_.max_dt_s) {
        time_discontinuity_ = true;
    }

    last_observation_.reset();
    last_result_ = tracker_->process(std::nullopt, dt);
    last_update_time_ = update_time;
    timestamp_warning_active_ = false;
}

void SuperPowerPredictor::clear() {
    resetTracker();
    last_observation_.reset();
    last_update_time_ = 0.0;
    last_dt_s_ = 0.0;
    update_frames_ = 0;
    debug_flip_flag_ = 1;
    has_update_time_ = false;
    timestamp_warning_active_ = false;
    time_discontinuity_ = false;
}

EKFTargetPrediction SuperPowerPredictor::predict(double predict_time) const {
    EKFTargetPrediction result;
    if (!hasState()) return result;

    const sp_ekf::Target* target = tracker_->target();
    if (!target) return result;

    Eigen::VectorXd x = target->ekfX();
    x[0] += x[1] * predict_time;
    x[2] += x[3] * predict_time;
    x[4] += x[5] * predict_time;
    x[6] = wrapAngle(x[6] + x[7] * predict_time);

    const double r1 = x[8];
    const double r2 = x[8] + x[9];
    const double h = x[10];

    result.center_x = x[0] * kMillimetersPerMeter;
    result.center_y = x[2] * kMillimetersPerMeter;
    result.center_z = x[4] * kMillimetersPerMeter;
    result.alternate_z = (x[4] + h) * kMillimetersPerMeter;
    result.r1 = r1 * kMillimetersPerMeter;
    result.r2 = r2 * kMillimetersPerMeter;
    result.h = h * kMillimetersPerMeter;
    result.yaw = toProjectYaw(x[6]);
    result.w = x[7];
    result.rotation_direction = x[7] >= 0.0 ? 1 : -1;

    const int armor_num = target->armorNum();
    result.armors.reserve(static_cast<std::size_t>(armor_num));
    for (int id = 0; id < armor_num; ++id) {
        const double angle = wrapAngle(
            x[6] + id * 2.0 * kPi / static_cast<double>(armor_num));
        const bool use_l_h = armor_num == 4 && (id == 1 || id == 3);
        const double radius = use_l_h ? r2 : r1;
        const double armor_x = x[0] - radius * std::cos(angle);
        const double armor_y = x[2] - radius * std::sin(angle);
        const double armor_z = use_l_h ? x[4] + h : x[4];

        result.armors.push_back(EKFPredictedArmor{
            armor_x * kMillimetersPerMeter,
            armor_y * kMillimetersPerMeter,
            armor_z * kMillimetersPerMeter,
            radius * kMillimetersPerMeter,
            toProjectYaw(angle),
        });
    }

    return result;
}

EKFTargetState SuperPowerPredictor::state() const {
    EKFTargetState result;
    if (!hasState()) return result;

    const sp_ekf::Target* target = tracker_->target();
    if (!target) return result;

    const Eigen::VectorXd x = target->ekfX();
    result.center_x = x[0] * kMillimetersPerMeter;
    result.center_vx = x[1] * kMillimetersPerMeter;
    result.center_y = x[2] * kMillimetersPerMeter;
    result.center_vy = x[3] * kMillimetersPerMeter;
    result.center_z = x[4] * kMillimetersPerMeter;
    result.center_vz = x[5] * kMillimetersPerMeter;
    result.yaw = toProjectYaw(x[6]);
    result.w = x[7];
    result.r1 = x[8] * kMillimetersPerMeter;
    result.r2 = (x[8] + x[9]) * kMillimetersPerMeter;
    result.h = x[10] * kMillimetersPerMeter;
    result.update_frames = update_frames_;
    return result;
}

EKFTargetDebugState SuperPowerPredictor::debugState() const {
    EKFTargetDebugState debug;
    debug.dt_s = last_dt_s_;
    debug.time_discontinuity = time_discontinuity_;
    debug.tracker_state = sp_ekf::trackerStateName(last_result_.state);
    debug.tracker_state_before =
        sp_ekf::trackerStateName(last_result_.state_before);
    debug.matched_id = last_result_.matched_id;
    debug.current_armor_id = last_result_.matched_id;
    debug.best_id = last_result_.matched_id;
    debug.measurement_valid = last_result_.measurement_valid;
    debug.updated = last_result_.updated;
    debug.lost_frames = last_result_.lost_frames;
    debug.nis = last_result_.nis;
    debug.position_error_m = last_result_.position_error;
    debug.yaw_error_deg = last_result_.angle_error >= 0.0
                              ? last_result_.angle_error * 180.0 / kPi
                              : -1.0;
    debug.armor_switched = last_result_.armor_switched;
    debug.candidate_is_switch = last_result_.armor_switched;
    debug.topology_event = last_result_.armor_switched;
    debug.geometry_update_allowed = last_result_.updated;
    debug.geometry_preserved = false;
    debug.geometry_valid = hasState();
    debug.armor_parity = last_result_.matched_id >= 0
                             ? last_result_.matched_id % 2
                             : -1;

    if (last_observation_) {
        debug.measurement << last_observation_->xyz[0],
                             last_observation_->xyz[1],
                             last_observation_->xyz[2],
                             toProjectYaw(last_observation_->angle);
        debug.measurement_yaw = debug.measurement[3];
    }

    if (last_result_.matched_id >= 0) {
        debug.pre_predicted << last_result_.predicted_xyza[0],
                               last_result_.predicted_xyza[1],
                               last_result_.predicted_xyza[2],
                               toProjectYaw(last_result_.predicted_xyza[3]);
        debug.predicted_yaw = debug.pre_predicted[3];
        if (last_observation_) {
            debug.pre_residual =
                last_observation_->xyz - last_result_.predicted_xyza.head<3>();
            debug.pre_position_error = debug.pre_residual.norm();
            debug.yaw_innovation = wrapAngle(
                debug.measurement_yaw - debug.predicted_yaw);
        }
    }

    const sp_ekf::Target* target = tracker_->target();
    if (hasState() && target) {
        const Eigen::VectorXd x = target->ekfX();
        const Eigen::MatrixXd& P = target->ekf().P;
        debug.r1_m = x[8];
        debug.r2_m = x[8] + x[9];
        debug.h_m = x[10];
        debug.p_r1_m2 = P(8, 8);
        debug.p_r2_m2 = P(8, 8) + P(9, 9) + 2.0 * P(8, 9);
        debug.p_h_m2 = P(10, 10);
        debug.p_x_m2 = P(0, 0);
        debug.p_vx_m2_s2 = P(1, 1);
        debug.p_y_m2 = P(2, 2);
        debug.p_vy_m2_s2 = P(3, 3);

        if (last_result_.matched_id >= 0) {
            const auto armors = target->armorXyzaList();
            const std::size_t id =
                static_cast<std::size_t>(last_result_.matched_id);
            if (id < armors.size()) {
                const Eigen::Vector4d& post = armors[id];
                debug.post_predicted << post[0], post[1], post[2],
                                        toProjectYaw(post[3]);
                if (last_observation_) {
                    debug.post_residual = last_observation_->xyz - post.head<3>();
                    debug.post_position_error = debug.post_residual.norm();
                }
            }
        }
    }

    return debug;
}

bool SuperPowerPredictor::ready() const {
    return tracker_ && tracker_->ready();
}

bool SuperPowerPredictor::hasState() const {
    return tracker_ && tracker_->hasState();
}

void SuperPowerPredictor::warnTimeIssue(const char* reason,
                                       double update_time,
                                       double dt) {
    if (!timestamp_warning_active_) {
        std::cerr << "[SuperPowerPredictor] warning: " << reason
                  << "; t=" << update_time << " dt=" << dt << " s"
                  << std::endl;
        timestamp_warning_active_ = true;
    }
}

void SuperPowerPredictor::resetTracker() {
    tracker_ = std::make_unique<sp_ekf::Tracker>(config_);
    last_result_ = sp_ekf::TrackerResult{};
}

void SuperPowerPredictor::initializeFromObservation(
    const EKFTargetObservation& observation) {
    if (!tracker_) resetTracker();
    last_observation_ = toSuperPower(observation);
    last_result_ = tracker_->process(last_observation_, 0.0);
    last_update_time_ = observation.t;
    last_dt_s_ = 0.0;
    has_update_time_ = true;
    update_frames_ = last_result_.initialized_this_frame ? 1 : 0;
    if (last_result_.matched_id >= 0) {
        debug_flip_flag_ = (last_result_.matched_id % 2) + 1;
    }
}

sp_ekf::ArmorObservation SuperPowerPredictor::toSuperPower(
    const EKFTargetObservation& observation) {
    sp_ekf::ArmorObservation result;
    result.xyz << observation.x / kMillimetersPerMeter,
                  observation.y / kMillimetersPerMeter,
                  observation.z / kMillimetersPerMeter;

    // Current project geometry uses p = c + r*[sin(yaw), -cos(yaw)].
    // SuperPower uses p = c - r*[cos(angle), sin(angle)].
    // angle = yaw + pi/2 makes the two definitions identical.
    result.angle = wrapAngle(observation.yaw + kHalfPi);
    return result;
}

double SuperPowerPredictor::toProjectYaw(double superpower_angle) {
    return wrapAngle(superpower_angle - kHalfPi);
}

double SuperPowerPredictor::wrapAngle(double angle) {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle <= -kPi) angle += 2.0 * kPi;
    return angle;
}
