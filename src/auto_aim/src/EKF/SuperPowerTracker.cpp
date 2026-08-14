#include "EKF/SuperPowerTracker.h"

#include <numeric>

namespace sp_ekf {

Tracker::Tracker(const TrackerConfig& config) : config_(config) {}

TrackerResult Tracker::process(
    const std::optional<ArmorObservation>& observation,
    double dt) {
    TrackerResult result;
    result.state_before = state_;
    result.measurement_valid = observation.has_value();

    // Match SuperPower: a large camera interval forces the tracker state to
    // LOST. The Target object itself can remain cached internally until the
    // next setTarget(), but no state is exposed while LOST.
    if (state_ != TrackerState::LOST && dt > config_.max_dt_s) {
        state_ = TrackerState::LOST;
    }

    bool found = false;
    TargetUpdateDebug update_debug;

    if (state_ == TrackerState::LOST) {
        if (observation) {
            found = setTarget(*observation);
            result.initialized_this_frame = found;
            result.updated = found;
            if (found) {
                result.matched_id = 0;
                result.predicted_xyza = target_->armorXyzaList().front();
            }
        }
    } else {
        if (observation) {
            update_debug = updateTarget(*observation, dt);
            found = true;
            result.updated = true;
            result.matched_id = update_debug.matched_id;
            result.armor_switched = update_debug.armor_switched;
            result.nis = update_debug.nis;
            result.position_error = update_debug.position_error;
            result.angle_error = update_debug.angle_error;
            result.predicted_xyza = update_debug.predicted_xyza;
        } else {
            predictOnly(dt);
            found = false;
        }
    }

    stateMachine(found);

    if (state_ != TrackerState::LOST && target_) {
        if (target_->diverged() || badConvergence()) {
            state_ = TrackerState::LOST;
        }
    }

    result.state = state_;
    result.lost_frames = temp_lost_count_;
    if (state_ == TrackerState::LOST && !target_) {
        result.matched_id = -1;
    }
    return result;
}

void Tracker::clear() {
    state_ = TrackerState::LOST;
    detect_count_ = 0;
    temp_lost_count_ = 0;
    target_.reset();
}

bool Tracker::setTarget(const ArmorObservation& observation) {
    // Generic normal 4-armor branch from SuperPower Tracker::set_target().
    target_.emplace(observation,
                    config_.initial_radius_m,
                    config_.armor_num,
                    normalFourArmorP0());
    return true;
}

TargetUpdateDebug Tracker::updateTarget(
    const ArmorObservation& observation,
    double dt) {
    target_->predict(dt);
    return target_->update(observation);
}

void Tracker::predictOnly(double dt) {
    if (target_) target_->predict(dt);
}

void Tracker::stateMachine(bool found) {
    if (state_ == TrackerState::LOST) {
        if (!found) return;
        state_ = TrackerState::DETECTING;
        detect_count_ = 1;
        return;
    }

    if (state_ == TrackerState::DETECTING) {
        if (found) {
            ++detect_count_;
            if (detect_count_ >= config_.min_detect_count) {
                state_ = TrackerState::TRACKING;
            }
        } else {
            detect_count_ = 0;
            state_ = TrackerState::LOST;
        }
        return;
    }

    if (state_ == TrackerState::TRACKING) {
        if (found) return;
        temp_lost_count_ = 1;
        state_ = TrackerState::TEMP_LOST;
        return;
    }

    if (state_ == TrackerState::TEMP_LOST) {
        if (found) {
            state_ = TrackerState::TRACKING;
            temp_lost_count_ = 0;
        } else {
            ++temp_lost_count_;
            if (temp_lost_count_ > config_.max_temp_lost_count) {
                state_ = TrackerState::LOST;
            }
        }
    }
}

bool Tracker::badConvergence() const {
    if (!target_) return false;
    const auto& failures = target_->ekf().recent_nis_failures;
    const int sum = std::accumulate(failures.begin(), failures.end(), 0);
    return sum >= static_cast<int>(0.4 * target_->ekf().window_size);
}

Eigen::VectorXd Tracker::normalFourArmorP0() {
    Eigen::VectorXd diag(11);
    diag << 1.0, 64.0,
            1.0, 64.0,
            1.0, 64.0,
            0.4, 100.0,
            1.0, 1.0, 1.0;
    return diag;
}

const char* trackerStateName(TrackerState state) {
    switch (state) {
        case TrackerState::LOST: return "LOST";
        case TrackerState::DETECTING: return "DETECTING";
        case TrackerState::TRACKING: return "TRACKING";
        case TrackerState::TEMP_LOST: return "TEMP_LOST";
    }
    return "LOST";
}

}  // namespace sp_ekf
