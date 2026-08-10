#pragma once

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "EKF/RobustArmorTracker.h"

struct EKFTargetObservation {
    double x;
    double y;
    double z;
    double yaw;
    double t;
};

struct EKFPredictedArmor {
    double x;
    double y;
    double z;
    double r;
    double yaw;
};

struct EKFTargetPrediction {
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double alternate_z = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    double h = 0.0;
    double yaw = 0.0;
    double w = 0.0;
    int rotation_direction = 1;
    std::vector<EKFPredictedArmor> armors;
};

struct EKFTargetState {
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double center_vx = 0.0;
    double center_vy = 0.0;
    double center_vz = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    double h = 0.0;
    double yaw = 0.0;
    double w = 0.0;
    unsigned long long update_frames = 0;
};

struct EKFTargetDebugState {
    std::string tracker_state = "LOST";
    int matched_id = -1;
    bool measurement_valid = false;
    bool updated = false;
    int lost_frames = 0;
    double nis = -1.0;
    double position_error_m = -1.0;
    double yaw_error_deg = -1.0;
    bool phase_observer_valid = false;
    double phase_w_instant = 0.0;
    double phase_w_filtered = 0.0;
    bool direction_reversal = false;
    bool armor_switched = false;
    bool recovered = false;
    bool phase_w_applied = false;
};

// Engineering adapter around the project's sole 3D target-motion backend.
// RobustArmorTracker owns all filtering, association and tracker-state behavior;
// this class only handles project units, timestamps, misses and result mapping.
class EKFTargetPredictor {
public:
    EKFTargetPredictor(const EKFTargetObservation& initial_observation,
                       double initial_radius_mm,
                       std::shared_ptr<YAML::Node> config_file_ptr);

    void update(const EKFTargetObservation& observation);
    void missUpdate(double update_time);

    EKFTargetPrediction predict(double predict_time) const;
    EKFTargetState state() const;
    EKFTargetDebugState debugState() const;

    bool ready() const;
    bool hasState() const;
    int debugFlipFlag() const { return debug_flip_flag_; }

private:
    static rm_ekf::ArmorObservation toMeters(
        const EKFTargetObservation& observation);

    std::unique_ptr<rm_ekf::RobustArmorTracker> tracker_;
    rm_ekf::TrackerResult last_result_;
    double last_update_time_ = 0.0;
    unsigned long long update_frames_ = 0;
    int debug_flip_flag_ = 1;
};
