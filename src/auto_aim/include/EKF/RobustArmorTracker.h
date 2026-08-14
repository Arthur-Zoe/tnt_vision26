#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <limits>
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
    // Continuous-time process noise. ArmorEKF discretizes these values using dt.
    double sigma_acc_xy = 3.0;
    double sigma_acc_z = 3.0;
    double sigma_angular_acc = 2.12132;
    double sigma_pos_rw_xy = 0.054006;
    double sigma_pos_rw_z = 0.054006;
    double sigma_yaw_rw = 0.054391;
    double sigma_radius_rw = 0.0;
    double sigma_height_rw = 0.0;

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
    int geometry_stable_frames_before_update = 5;
    bool freeze_radius_kalman_rows = false;
    bool statistically_fixed_radius = false;
    double fixed_radius_variance = 1e-8;
    bool geometry_reinit_covariance_floor_enabled = false;
    double geometry_reinit_radius_variance_floor = 0.0;
    double geometry_reinit_height_variance_floor = 0.0;
    bool association_debug_enable = false;

    ArmorState initial_filter;

    static RobustTrackerConfig fromYaml(const YAML::Node& root, double init_radius_m);
};

struct AssociationCandidate {
    int armor_id = -1;
    double yaw_variance_scale = 1.0;
    Eigen::Matrix<double, 4, 1> predicted = Eigen::Matrix<double, 4, 1>::Zero();
    Eigen::Matrix<double, 4, 1> innovation = Eigen::Matrix<double, 4, 1>::Zero();
    double nis = 1e30;
    double position_error = 1e30;
    double yaw_error = 1e30;
    Eigen::Matrix<double, 4, 1> nis_contribution =
        Eigen::Matrix<double, 4, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    bool numerically_valid = false;
};

struct AssociationHypothesisDebug {
    int armor_id = -1;
    ArmorObservation predicted;
    double facing_angle = std::numeric_limits<double>::quiet_NaN();
    bool range_pass = false;
    bool visibility_pass = false;
    AssociationCandidate measurement;
    AssociationCandidate hypothetical_scaled_yaw_measurement;
    // Diagnostic only: same state/innovation as `measurement`, but with the
    // radius covariance rows/columns isolated at the configured fixed value.
    // This candidate must never participate in association or EKF updates.
    AssociationCandidate statistically_fixed_radius_measurement;
    double radial_residual = std::numeric_limits<double>::quiet_NaN();
    double tangential_residual = std::numeric_limits<double>::quiet_NaN();
    bool nis_gate_pass = false;
    bool position_gate_pass = false;
    bool yaw_gate_pass = false;
    bool passes_all_measurement_gates = false;
};

class ArmorModel {
public:
    static ArmorObservation getArmor(const ArmorState& s, int armor_id,
                                     double predict_time = 0.0);
    static std::vector<ArmorObservation> getArmors(
        const ArmorState& s, double predict_time = 0.0);
    static double facingScore(const ArmorState& s,
                              const ArmorObservation& armor,
                              double predict_time = 0.0);
    static Eigen::Matrix<double, 4, 1> measurementFunction(
        const Eigen::Matrix<double, 11, 1>& X, int armor_id);
    static Eigen::Matrix<double, 4, 11> measurementJacobian(
        const Eigen::Matrix<double, 11, 1>& X, int armor_id);
};

class ArmorEKF {
public:
    void configure(const RobustTrackerConfig& cfg);
    void reset(const ArmorState& initial_state);
    void invalidate();

    bool initialized() const { return initialized_; }
    ArmorState state() const { return ArmorState::fromVector(X_); }

    void predict(double dt);
    void update(const Eigen::Matrix<double, 4, 1>& z,
                int armor_id,
                double yaw_variance_scale = 1.0,
                bool update_geometry = true);
    void updateAngularVelocity(double measured_w, double variance);
    AssociationCandidate evaluateMeasurement(
        const Eigen::Matrix<double, 4, 1>& z,
        int armor_id,
        double yaw_variance_scale = 1.0) const;
    AssociationCandidate evaluateMeasurementWithFixedRadiusCovariance(
        const Eigen::Matrix<double, 4, 1>& z,
        int armor_id,
        double yaw_variance_scale = 1.0) const;
    std::array<double, 3> geometryVariances() const;
    std::array<double, 11> stateVariances() const;
    Eigen::Matrix<double, 11, 11> covariance() const { return P_; }
    void setGeometryVariances(double var_r1, double var_r2, double var_h);

private:
    void enforcePhysicalLimits();
    void enforceRadiusCovarianceIsolation();

    bool initialized_ = false;
    Eigen::Matrix<double, 11, 1> X_ = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> P_ = Eigen::Matrix<double, 11, 11>::Identity();
    Eigen::Matrix<double, 4, 4> R_ = Eigen::Matrix<double, 4, 4>::Identity();

    double sigma_acc_xy_ = 3.0;
    double sigma_acc_z_ = 3.0;
    double sigma_angular_acc_ = 2.12132;
    double sigma_pos_rw_xy_ = 0.054006;
    double sigma_pos_rw_z_ = 0.054006;
    double sigma_yaw_rw_ = 0.054391;
    double sigma_radius_rw_ = 0.0;
    double sigma_height_rw_ = 0.0;

    double min_radius_ = 0.08;
    double max_radius_ = 0.60;
    double max_abs_h_ = 0.40;
    double max_abs_w_ = 20.0;
    bool freeze_radius_kalman_rows_ = false;
    bool statistically_fixed_radius_ = false;
    double fixed_radius_variance_ = 1e-8;
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
    TrackerState tracker_state_before = TrackerState::LOST;
    int matched_id = -1;
    bool measurement_valid = false;
    bool initialized_this_frame = false;
    bool updated = false;
    bool recovered = false;
    bool temp_lost_recovery = false;
    bool armor_switched = false;
    bool candidate_is_switch = false;
    bool topology_event = false;
    int lost_frames = 0;
    int detect_count = 0;
    double nis = -1.0;
    double position_error = -1.0;
    double yaw_error = -1.0;
    int best_id = -1;
    double measurement_yaw = std::numeric_limits<double>::quiet_NaN();
    double predicted_yaw = std::numeric_limits<double>::quiet_NaN();
    double yaw_innovation = std::numeric_limits<double>::quiet_NaN();
    Eigen::Matrix<double, 4, 1> measurement =
        Eigen::Matrix<double, 4, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix<double, 4, 1> pre_predicted =
        Eigen::Matrix<double, 4, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix<double, 4, 1> post_predicted =
        Eigen::Matrix<double, 4, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix<double, 3, 1> pre_residual =
        Eigen::Matrix<double, 3, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix<double, 3, 1> post_residual =
        Eigen::Matrix<double, 3, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    double pre_position_error = std::numeric_limits<double>::quiet_NaN();
    double post_position_error = std::numeric_limits<double>::quiet_NaN();
    double residual_radial = std::numeric_limits<double>::quiet_NaN();
    double residual_tangential = std::numeric_limits<double>::quiet_NaN();
    double nis_xyz = std::numeric_limits<double>::quiet_NaN();
    double nis_yaw = std::numeric_limits<double>::quiet_NaN();
    double yaw_variance_scale = 1.0;
    double hypothetical_scaled_nis = std::numeric_limits<double>::quiet_NaN();
    Eigen::Matrix<double, 4, 1> hypothetical_scaled_nis_contribution =
        Eigen::Matrix<double, 4, 1>::Constant(
            std::numeric_limits<double>::quiet_NaN());

    bool phase_observer_valid = false;
    double phase_delta = 0.0;
    double phase_w_instant = 0.0;
    double phase_w_filtered = 0.0;
    bool direction_reversal = false;
    bool pending_sign_conflict = false;
    bool phase_w_applied = false;

    bool geometry_valid = false;
    bool geometry_update_allowed = false;
    bool geometry_preserved = false;

    int current_armor_id = -1;
    std::array<AssociationHypothesisDebug, 4> association_hypotheses;
};

struct GeometryMemory {
    double r1 = 0.0;
    double r2 = 0.0;
    double h = 0.0;
    double var_r1 = 0.0;
    double var_r2 = 0.0;
    double var_h = 0.0;
    bool valid = false;
};

class RobustArmorTracker {
public:
    void configure(const RobustTrackerConfig& cfg);
    void loseTrackPreserveGeometry();
    void clear();

    TrackerResult process(const std::optional<ArmorObservation>& measurement,
                          double dt,
                          int forced_physical_armor_id = -1);

    bool ready() const { return state_ == TrackerState::TRACKING; }
    bool hasState() const { return ekf_.initialized(); }
    TrackerState trackerState() const { return state_; }
    int currentArmorId() const { return current_armor_id_; }
    ArmorState state() const { return ekf_.state(); }
    std::array<double, 3> geometryVariances() const {
        return ekf_.geometryVariances();
    }
    std::array<double, 11> stateVariances() const {
        return ekf_.stateVariances();
    }
    const GeometryMemory& geometryMemory() const { return geometry_memory_; }
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
                                           bool phase_conflict,
                                           bool* armor_switched,
                                           std::array<AssociationHypothesisDebug, 4>*
                                               debug_hypotheses) const;
    TrackerResult handleMiss(TrackerResult result);
    PhaseUpdate observeRotationPhase(const ArmorObservation& obs, double dt);
    void resetPhaseObserver();
    void reset(const ArmorState& state, int physical_armor_id, TrackerState tracker_state);
    bool geometryStateValid(const ArmorState& state) const;
    void updateGeometryMemory();
    void populateGeometryDebug(TrackerResult& result) const;

    RobustTrackerConfig cfg_;
    ArmorEKF ekf_;
    TrackerState state_ = TrackerState::LOST;
    int current_armor_id_ = -1;
    int detect_count_ = 0;
    int detect_misses_ = 0;
    int lost_frames_ = 0;
    int stable_association_frames_ = 0;
    GeometryMemory geometry_memory_;

    bool have_phase_yaw_ = false;
    double last_phase_yaw_ = 0.0;
    double phase_elapsed_ = 0.0;
    bool phase_w_valid_ = false;
    double phase_w_filtered_ = 0.0;
    int reversal_conflict_count_ = 0;
    double reversal_conflict_sum_ = 0.0;
};

}  // namespace rm_ekf
