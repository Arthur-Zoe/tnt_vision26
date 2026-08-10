#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace rm_ekf {

constexpr double kPi = 3.14159265358979323846;

inline double wrapAngle(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a < -kPi) a += 2.0 * kPi;
    return a;
}

inline double deg2rad(double deg) { return deg * kPi / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / kPi; }

struct ArmorState {
    // Internal robust-EKF units: meter, second, radian.
    // [x, vx, y, vy, z, vz, yaw, w, r1, r2, h]
    // x:right, y:forward, z:up
    // armor = center + [r*sin(yaw), -r*cos(yaw)]
    double x = 0.0;
    double vx = 0.0;
    double y = 3.0;
    double vy = 0.0;
    double z = 0.5;
    double vz = 0.0;
    double yaw = 0.0;
    double w = 0.0;
    double r1 = 0.25;
    double r2 = 0.25;
    double h = 0.0;

    Eigen::Matrix<double, 11, 1> toVector() const;
    static ArmorState fromVector(const Eigen::Matrix<double, 11, 1>& X);
};

struct ArmorObservation {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
    int id = -1;

    Eigen::Matrix<double, 4, 1> toVector() const;
};

struct RobustTrackerConfig {
    // Same defaults as the validated v4 replay project.
    double q_pos = 1e-4;
    double q_vel = 1e-2;
    double q_z = 1e-4;
    double q_vz = 1e-2;
    double q_yaw = 1e-4;
    double q_w = 5e-3;
    double q_radius = 1e-6;
    double q_h = 1e-6;

    double r_pos = 2e-3;
    double r_z = 2e-3;
    double r_yaw = 2e-3;

    double state_min_radius = 0.08;
    double state_max_radius = 0.60;
    double state_max_abs_h = 0.40;
    double state_max_abs_w = 20.0;

    int detect_confirm_frames = 3;
    int detect_max_misses = 2;
    int temp_lost_max_frames = 12;

    double association_nis_gate = 13.28;
    double association_switch_nis_margin = 1.50;
    double association_max_position_error = 0.80;
    double association_max_yaw_error_deg = 65.0;

    bool rotation_phase_observer_enable = true;
    double rotation_phase_alpha = 0.45;
    double rotation_phase_max_abs_omega = 15.0;
    double rotation_phase_max_step_deg = 35.0;
    double rotation_phase_min_dt = 0.005;
    double rotation_phase_max_dt = 0.20;
    double rotation_phase_w_variance = 0.30;
    double rotation_reversal_min_abs_omega = 0.60;
    int rotation_reversal_confirm_frames = 2;
    double rotation_reversal_w_variance = 0.005;
    double rotation_switch_yaw_r_scale = 9.0;

    double input_min_range = 0.20;
    double input_max_range = 8.00;
    double input_max_abs_z = 2.50;

    double armor_visible_angle_deg = 70.0;

    ArmorState initial_filter;

    static RobustTrackerConfig fromYaml(const YAML::Node& root, double init_radius_m);
};

struct AssociationCandidate {
    int armor_id = -1;
    Eigen::Matrix<double, 4, 1> predicted = Eigen::Matrix<double, 4, 1>::Zero();
    Eigen::Matrix<double, 4, 1> innovation = Eigen::Matrix<double, 4, 1>::Zero();
    double nis = 1e30;
    double position_error = 1e30;
    double yaw_error = 1e30;
    bool numerically_valid = false;
};

class ArmorModel {
public:
    static ArmorObservation getArmor(const ArmorState& s, int armor_id, double predict_time = 0.0);
    static std::vector<ArmorObservation> getArmors(const ArmorState& s, double predict_time = 0.0);
    static double facingScore(const ArmorState& s, const ArmorObservation& armor, double predict_time = 0.0);
    static Eigen::Matrix<double, 4, 1> measurementFunction(const Eigen::Matrix<double, 11, 1>& X, int armor_id);
    static Eigen::Matrix<double, 4, 11> measurementJacobian(const Eigen::Matrix<double, 11, 1>& X, int armor_id);
};

class ArmorEKF {
public:
    void configure(const RobustTrackerConfig& cfg);
    void reset(const ArmorState& initial_state);
    void invalidate();

    bool initialized() const { return initialized_; }
    ArmorState state() const { return ArmorState::fromVector(X_); }

    void predict(double dt);
    void update(const Eigen::Matrix<double, 4, 1>& z, int armor_id, double yaw_variance_scale = 1.0);
    void updateAngularVelocity(double measured_w, double variance);
    AssociationCandidate evaluateMeasurement(const Eigen::Matrix<double, 4, 1>& z, int armor_id) const;

private:
    void enforcePhysicalLimits();

    bool initialized_ = false;
    Eigen::Matrix<double, 11, 1> X_ = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> P_ = Eigen::Matrix<double, 11, 11>::Identity();
    Eigen::Matrix<double, 11, 11> Q_ = Eigen::Matrix<double, 11, 11>::Identity();
    Eigen::Matrix<double, 4, 4> R_ = Eigen::Matrix<double, 4, 4>::Identity();

    double min_radius_ = 0.08;
    double max_radius_ = 0.60;
    double max_abs_h_ = 0.40;
    double max_abs_w_ = 20.0;
};

enum class TrackerState {
    LOST = 0,
    DETECTING,
    TRACKING,
    TEMP_LOST
};

const char* trackerStateName(TrackerState state);

struct TrackerResult {
    TrackerState state = TrackerState::LOST;
    int matched_id = -1;
    bool measurement_valid = false;
    bool initialized_this_frame = false;
    bool updated = false;
    bool recovered = false;
    bool armor_switched = false;
    int lost_frames = 0;
    int detect_count = 0;
    double nis = -1.0;
    double position_error = -1.0;
    double yaw_error = -1.0;

    bool phase_observer_valid = false;
    double phase_delta = 0.0;
    double phase_w_instant = 0.0;
    double phase_w_filtered = 0.0;
    bool direction_reversal = false;
    bool phase_w_applied = false;
};

class RobustArmorTracker {
public:
    void configure(const RobustTrackerConfig& cfg);
    void clear();

    TrackerResult process(const std::optional<ArmorObservation>& measurement,
                          double dt,
                          int forced_physical_armor_id = -1);

    bool ready() const { return state_ == TrackerState::TRACKING; }
    bool hasState() const { return ekf_.initialized(); }
    TrackerState trackerState() const { return state_; }
    int currentArmorId() const { return current_armor_id_; }
    ArmorState state() const { return ekf_.state(); }
    std::vector<ArmorObservation> predictArmors(double predict_time) const {
        return ArmorModel::getArmors(ekf_.state(), predict_time);
    }

private:
    struct PhaseUpdate {
        bool valid = false;
        double delta = 0.0;
        double instant_w = 0.0;
        double filtered_w = 0.0;
        bool reversal_confirmed = false;
        bool pending_sign_conflict = false;
    };

    bool validInput(const ArmorObservation& obs) const;
    ArmorState initializeStateFromMeasurement(const ArmorObservation& obs, int armor_id) const;
    AssociationCandidate chooseAssociation(const Eigen::Matrix<double, 4, 1>& z,
                                           int forced_physical_armor_id,
                                           bool* armor_switched) const;
    TrackerResult handleMiss(TrackerResult result);
    PhaseUpdate observeRotationPhase(const ArmorObservation& obs, double dt);
    void resetPhaseObserver();
    void reset(const ArmorState& state, int physical_armor_id, TrackerState tracker_state);

    RobustTrackerConfig cfg_;
    ArmorEKF ekf_;
    TrackerState state_ = TrackerState::LOST;
    int current_armor_id_ = -1;
    int detect_count_ = 0;
    int detect_misses_ = 0;
    int lost_frames_ = 0;

    bool have_phase_yaw_ = false;
    double last_phase_yaw_ = 0.0;
    double phase_elapsed_ = 0.0;
    bool phase_w_valid_ = false;
    double phase_w_filtered_ = 0.0;
    int reversal_conflict_count_ = 0;
    double reversal_conflict_sum_ = 0.0;
};

}  // namespace rm_ekf
