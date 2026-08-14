#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include "EKF/SuperPowerTarget.h"

namespace sp_ekf {

enum class TrackerState {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
};

struct TrackerConfig {
    // SuperPower standard3.yaml defaults.
    int min_detect_count = 5;
    int max_temp_lost_count = 15;
    double max_dt_s = 0.1;
    double initial_radius_m = 0.2;
    int armor_num = 4;
};

struct TrackerResult {
    TrackerState state = TrackerState::LOST;
    TrackerState state_before = TrackerState::LOST;
    bool initialized_this_frame = false;
    bool measurement_valid = false;
    bool updated = false;
    int lost_frames = 0;
    int matched_id = -1;
    bool armor_switched = false;
    double nis = -1.0;
    double position_error = -1.0;
    double angle_error = -1.0;
    Eigen::Vector4d predicted_xyza = Eigen::Vector4d::Zero();
};

class Tracker {
public:
    explicit Tracker(const TrackerConfig& config = TrackerConfig{});

    TrackerResult process(const std::optional<ArmorObservation>& observation,
                          double dt);
    void clear();

    // SuperPower keeps the last Target object internally after entering LOST,
    // but does not return it to downstream modules. Mirror that observable
    // behavior here: LOST means no usable state.
    bool hasState() const {
        return state_ != TrackerState::LOST && target_.has_value();
    }
    bool ready() const {
        return state_ == TrackerState::TRACKING && target_.has_value();
    }
    TrackerState state() const { return state_; }
    const Target* target() const {
        return target_ ? &(*target_) : nullptr;
    }

private:
    TrackerConfig config_;
    int detect_count_ = 0;
    int temp_lost_count_ = 0;
    TrackerState state_ = TrackerState::LOST;
    std::optional<Target> target_;

    bool setTarget(const ArmorObservation& observation);
    TargetUpdateDebug updateTarget(const ArmorObservation& observation,
                                   double dt);
    void predictOnly(double dt);
    void stateMachine(bool found);
    bool badConvergence() const;

    static Eigen::VectorXd normalFourArmorP0();
};

const char* trackerStateName(TrackerState state);

}  // namespace sp_ekf
