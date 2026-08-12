#include "EKF/RobustArmorTracker.h"

#include <algorithm>
#include <limits>

namespace rm_ekf {

namespace {

template <typename T>
void readIfExists(const YAML::Node& node, const char* key, T& value) {
    if (node && node[key]) value = node[key].as<T>();
}

}  // namespace

Eigen::Matrix<double, 11, 1> ArmorState::toVector() const {
    Eigen::Matrix<double, 11, 1> X;
    X << x, vx, y, vy, z, vz, yaw, w, r1, r2, h;
    return X;
}

ArmorState ArmorState::fromVector(const Eigen::Matrix<double, 11, 1>& X) {
    ArmorState s;
    s.x = X(0);   s.vx = X(1);
    s.y = X(2);   s.vy = X(3);
    s.z = X(4);   s.vz = X(5);
    s.yaw = wrapAngle(X(6));
    s.w = X(7);
    s.r1 = X(8);  s.r2 = X(9);
    s.h = X(10);
    return s;
}

Eigen::Matrix<double, 4, 1> ArmorObservation::toVector() const {
    Eigen::Matrix<double, 4, 1> zvec;
    zvec << x, y, z, yaw;
    return zvec;
}

RobustTrackerConfig RobustTrackerConfig::fromYaml(const YAML::Node& root, double init_radius_m) {
    RobustTrackerConfig cfg;
    cfg.initial_filter.r1 = init_radius_m;
    cfg.initial_filter.r2 = init_radius_m;
    cfg.initial_filter.h = 0.0;

    const YAML::Node n = root["robust_ekf"];
    if (!n) return cfg;

    // Keep the online initialization geometry identical to the validated
    // standalone v4 replay unless explicitly overridden for a robot type.
    readIfExists(n, "initial_r1", cfg.initial_filter.r1);
    readIfExists(n, "initial_r2", cfg.initial_filter.r2);
    readIfExists(n, "initial_h", cfg.initial_filter.h);

    const YAML::Node process_noise = n["process_noise"];
    readIfExists(process_noise, "sigma_acc_xy", cfg.sigma_acc_xy);
    readIfExists(process_noise, "sigma_acc_z", cfg.sigma_acc_z);
    readIfExists(process_noise, "sigma_angular_acc", cfg.sigma_angular_acc);
    readIfExists(process_noise, "sigma_pos_rw_xy", cfg.sigma_pos_rw_xy);
    readIfExists(process_noise, "sigma_pos_rw_z", cfg.sigma_pos_rw_z);
    readIfExists(process_noise, "sigma_yaw_rw", cfg.sigma_yaw_rw);
    readIfExists(process_noise, "sigma_radius_rw", cfg.sigma_radius_rw);
    readIfExists(process_noise, "sigma_height_rw", cfg.sigma_height_rw);

    readIfExists(n, "r_pos", cfg.r_pos);
    readIfExists(n, "r_z", cfg.r_z);
    readIfExists(n, "r_yaw", cfg.r_yaw);

    readIfExists(n, "state_min_radius", cfg.state_min_radius);
    readIfExists(n, "state_max_radius", cfg.state_max_radius);
    readIfExists(n, "state_max_abs_h", cfg.state_max_abs_h);
    readIfExists(n, "state_max_abs_w", cfg.state_max_abs_w);

    readIfExists(n, "detect_confirm_frames", cfg.detect_confirm_frames);
    readIfExists(n, "detect_max_misses", cfg.detect_max_misses);
    readIfExists(n, "temp_lost_max_frames", cfg.temp_lost_max_frames);

    readIfExists(n, "association_nis_gate", cfg.association_nis_gate);
    readIfExists(n, "association_switch_nis_margin", cfg.association_switch_nis_margin);
    readIfExists(n, "association_max_position_error", cfg.association_max_position_error);
    readIfExists(n, "association_max_yaw_error_deg", cfg.association_max_yaw_error_deg);

    readIfExists(n, "rotation_phase_observer_enable", cfg.rotation_phase_observer_enable);
    readIfExists(n, "rotation_phase_alpha", cfg.rotation_phase_alpha);
    readIfExists(n, "rotation_phase_max_abs_omega", cfg.rotation_phase_max_abs_omega);
    readIfExists(n, "rotation_phase_max_step_deg", cfg.rotation_phase_max_step_deg);
    readIfExists(n, "rotation_phase_min_dt", cfg.rotation_phase_min_dt);
    readIfExists(n, "rotation_phase_max_dt", cfg.rotation_phase_max_dt);
    readIfExists(n, "rotation_phase_w_variance", cfg.rotation_phase_w_variance);
    readIfExists(n, "rotation_reversal_min_abs_omega", cfg.rotation_reversal_min_abs_omega);
    readIfExists(n, "rotation_reversal_confirm_frames", cfg.rotation_reversal_confirm_frames);
    readIfExists(n, "rotation_reversal_w_variance", cfg.rotation_reversal_w_variance);
    readIfExists(n, "rotation_switch_yaw_r_scale", cfg.rotation_switch_yaw_r_scale);

    readIfExists(n, "input_min_range", cfg.input_min_range);
    readIfExists(n, "input_max_range", cfg.input_max_range);
    readIfExists(n, "input_max_abs_z", cfg.input_max_abs_z);
    readIfExists(n, "armor_visible_angle_deg", cfg.armor_visible_angle_deg);

    const YAML::Node geometry = n["geometry"];
    readIfExists(geometry, "stable_frames_before_update",
                 cfg.geometry_stable_frames_before_update);
    const YAML::Node reinit_covariance_floor =
        geometry["reinit_covariance_floor"];
    readIfExists(reinit_covariance_floor, "enabled",
                 cfg.geometry_reinit_covariance_floor_enabled);
    readIfExists(reinit_covariance_floor, "radius_variance",
                 cfg.geometry_reinit_radius_variance_floor);
    readIfExists(reinit_covariance_floor, "height_variance",
                 cfg.geometry_reinit_height_variance_floor);
    readIfExists(geometry, "association_debug_csv",
                 cfg.association_debug_enable);

    return cfg;
}

ArmorObservation ArmorModel::getArmor(
    const ArmorState& s, int armor_id, double predict_time) {
    const int i = ((armor_id % 4) + 4) % 4;
    const double cx = s.x + s.vx * predict_time;
    const double cy = s.y + s.vy * predict_time;
    const double cz = s.z + s.vz * predict_time;
    const double yaw = wrapAngle(s.yaw + s.w * predict_time + i * kPi / 2.0);
    const double r = (i % 2 == 0) ? s.r1 : s.r2;
    const double dz = (i % 2 == 0) ? 0.0 : s.h;

    ArmorObservation obs;
    obs.x = cx + r * std::sin(yaw);
    obs.y = cy - r * std::cos(yaw);
    obs.z = cz + dz;
    obs.yaw = yaw;
    obs.id = i;
    return obs;
}

std::vector<ArmorObservation> ArmorModel::getArmors(
    const ArmorState& s, double predict_time) {
    std::vector<ArmorObservation> armors;
    armors.reserve(4);
    for (int i = 0; i < 4; ++i) {
        armors.push_back(getArmor(s, i, predict_time));
    }
    return armors;
}

double ArmorModel::facingScore(const ArmorState& s, const ArmorObservation& armor, double predict_time) {
    const double cx = s.x + s.vx * predict_time;
    const double cy = s.y + s.vy * predict_time;
    const double cam_to_center_yaw = std::atan2(-cx, cy);
    return std::abs(wrapAngle(armor.yaw - cam_to_center_yaw));
}

Eigen::Matrix<double, 4, 1> ArmorModel::measurementFunction(
    const Eigen::Matrix<double, 11, 1>& X, int armor_id) {
    const int i = ((armor_id % 4) + 4) % 4;
    const double x = X(0);
    const double y = X(2);
    const double z = X(4);
    const double yaw = wrapAngle(X(6) + i * kPi / 2.0);
    const double r = (i % 2 == 0) ? X(8) : X(9);
    const double dz = (i % 2 == 0) ? 0.0 : X(10);

    Eigen::Matrix<double, 4, 1> h;
    h << x + r * std::sin(yaw),
         y - r * std::cos(yaw),
         z + dz,
         yaw;
    return h;
}

Eigen::Matrix<double, 4, 11> ArmorModel::measurementJacobian(
    const Eigen::Matrix<double, 11, 1>& X, int armor_id) {
    const int i = ((armor_id % 4) + 4) % 4;
    const double yaw = wrapAngle(X(6) + i * kPi / 2.0);
    const double r = (i % 2 == 0) ? X(8) : X(9);

    Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
    H(0, 0) = 1.0;
    H(0, 6) = r * std::cos(yaw);
    if (i % 2 == 0) H(0, 8) = std::sin(yaw);
    else            H(0, 9) = std::sin(yaw);

    H(1, 2) = 1.0;
    H(1, 6) = r * std::sin(yaw);
    if (i % 2 == 0) H(1, 8) = -std::cos(yaw);
    else            H(1, 9) = -std::cos(yaw);

    H(2, 4) = 1.0;
    if (i % 2 == 1) H(2, 10) = 1.0;

    H(3, 6) = 1.0;
    return H;
}

void ArmorEKF::configure(const RobustTrackerConfig& cfg) {
    sigma_acc_xy_ = cfg.sigma_acc_xy;
    sigma_acc_z_ = cfg.sigma_acc_z;
    sigma_angular_acc_ = cfg.sigma_angular_acc;
    sigma_pos_rw_xy_ = cfg.sigma_pos_rw_xy;
    sigma_pos_rw_z_ = cfg.sigma_pos_rw_z;
    sigma_yaw_rw_ = cfg.sigma_yaw_rw;
    sigma_radius_rw_ = cfg.sigma_radius_rw;
    sigma_height_rw_ = cfg.sigma_height_rw;

    R_.setZero();
    R_(0, 0) = cfg.r_pos;
    R_(1, 1) = cfg.r_pos;
    R_(2, 2) = cfg.r_z;
    R_(3, 3) = cfg.r_yaw;

    min_radius_ = cfg.state_min_radius;
    max_radius_ = cfg.state_max_radius;
    max_abs_h_ = cfg.state_max_abs_h;
    max_abs_w_ = cfg.state_max_abs_w;
}

void ArmorEKF::reset(const ArmorState& initial_state) {
    X_ = initial_state.toVector();
    P_.setIdentity();
    P_ *= 0.05;
    P_(1, 1) = 0.50;
    P_(3, 3) = 0.50;
    P_(5, 5) = 0.50;
    P_(7, 7) = 0.80;
    P_(8, 8) = 0.02;
    P_(9, 9) = 0.02;
    P_(10, 10) = 0.02;
    enforcePhysicalLimits();
    initialized_ = true;
}

void ArmorEKF::invalidate() {
    initialized_ = false;
}

void ArmorEKF::enforcePhysicalLimits() {
    X_(6) = wrapAngle(X_(6));
    X_(7) = std::clamp(X_(7), -max_abs_w_, max_abs_w_);
    X_(8) = std::clamp(X_(8), min_radius_, max_radius_);
    X_(9) = std::clamp(X_(9), min_radius_, max_radius_);
    X_(10) = std::clamp(X_(10), -max_abs_h_, max_abs_h_);
}

void ArmorEKF::predict(double dt) {
    if (!initialized_ || !(dt > 0.0) || !std::isfinite(dt)) return;

    Eigen::Matrix<double, 11, 11> F = Eigen::Matrix<double, 11, 11>::Identity();
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    Eigen::Matrix<double, 11, 11> Q =
        Eigen::Matrix<double, 11, 11>::Zero();
    const auto set_acceleration_pair =
        [&Q, dt2, dt3, dt4](int position_index, int velocity_index,
                            double sigma) {
            const double sigma2 = sigma * sigma;
            Q(position_index, position_index) = 0.25 * dt4 * sigma2;
            Q(position_index, velocity_index) = 0.5 * dt3 * sigma2;
            Q(velocity_index, position_index) = 0.5 * dt3 * sigma2;
            Q(velocity_index, velocity_index) = dt2 * sigma2;
        };
    set_acceleration_pair(0, 1, sigma_acc_xy_);
    set_acceleration_pair(2, 3, sigma_acc_xy_);
    set_acceleration_pair(4, 5, sigma_acc_z_);
    set_acceleration_pair(6, 7, sigma_angular_acc_);
    Q(0, 0) += sigma_pos_rw_xy_ * sigma_pos_rw_xy_ * dt;
    Q(2, 2) += sigma_pos_rw_xy_ * sigma_pos_rw_xy_ * dt;
    Q(4, 4) += sigma_pos_rw_z_ * sigma_pos_rw_z_ * dt;
    Q(6, 6) += sigma_yaw_rw_ * sigma_yaw_rw_ * dt;
    Q(8, 8) = sigma_radius_rw_ * sigma_radius_rw_ * dt;
    Q(9, 9) = sigma_radius_rw_ * sigma_radius_rw_ * dt;
    Q(10, 10) = sigma_height_rw_ * sigma_height_rw_ * dt;

    X_ = F * X_;
    enforcePhysicalLimits();
    P_ = F * P_ * F.transpose() + Q;
    P_ = 0.5 * (P_ + P_.transpose());
}

AssociationCandidate ArmorEKF::evaluateMeasurement(
    const Eigen::Matrix<double, 4, 1>& z,
    int armor_id,
    double yaw_variance_scale) const {
    AssociationCandidate c;
    c.armor_id = ((armor_id % 4) + 4) % 4;
    c.yaw_variance_scale = std::max(1.0, yaw_variance_scale);
    if (!initialized_) return c;

    c.predicted = ArmorModel::measurementFunction(X_, c.armor_id);
    const Eigen::Matrix<double, 4, 11> H =
        ArmorModel::measurementJacobian(X_, c.armor_id);

    c.innovation = z - c.predicted;
    c.innovation(3) = wrapAngle(c.innovation(3));
    c.position_error = c.innovation.head<3>().norm();
    c.yaw_error = std::abs(c.innovation(3));

    Eigen::Matrix<double, 4, 4> R_eff = R_;
    R_eff(3, 3) *= c.yaw_variance_scale;
    const Eigen::Matrix<double, 4, 4> S = H * P_ * H.transpose() + R_eff;
    const auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success) return c;

    const Eigen::Matrix<double, 4, 1> solved = ldlt.solve(c.innovation);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) return c;

    c.nis = c.innovation.dot(solved);
    c.nis_contribution = c.innovation.cwiseProduct(solved);
    c.numerically_valid = std::isfinite(c.nis) && c.nis >= 0.0;
    return c;
}

void ArmorEKF::update(const Eigen::Matrix<double, 4, 1>& z,
                      int armor_id,
                      double yaw_variance_scale,
                      bool update_geometry) {
    if (!initialized_) return;

    const int id = ((armor_id % 4) + 4) % 4;
    const Eigen::Matrix<double, 4, 1> h =
        ArmorModel::measurementFunction(X_, id);
    const Eigen::Matrix<double, 4, 11> H =
        ArmorModel::measurementJacobian(X_, id);

    Eigen::Matrix<double, 4, 1> innovation = z - h;
    innovation(3) = wrapAngle(innovation(3));

    Eigen::Matrix<double, 4, 4> R_eff = R_;
    R_eff(3, 3) *= yaw_variance_scale;
    const Eigen::Matrix<double, 4, 4> S = H * P_ * H.transpose() + R_eff;
    const auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success) return;

    const Eigen::Matrix<double, 11, 4> PHt = P_ * H.transpose();
    const Eigen::Matrix<double, 4, 11> solved = ldlt.solve(PHt.transpose());
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) return;
    Eigen::Matrix<double, 11, 4> K = solved.transpose();

    // Geometry is static over predict(). On topology/recovery events, keep the
    // normal motion/yaw correction but prevent the same innovation from
    // rewriting geometry. During stable updates, apply the minimum parity
    // observability mask: EVEN updates r1; ODD updates r2 and h.
    if (!update_geometry) {
        K.row(8).setZero();
        K.row(9).setZero();
        K.row(10).setZero();
    } else if (id % 2 == 0) {
        K.row(9).setZero();
        K.row(10).setZero();
    } else {
        K.row(8).setZero();
    }

    X_ += K * innovation;
    enforcePhysicalLimits();

    const Eigen::Matrix<double, 11, 11> I = Eigen::Matrix<double, 11, 11>::Identity();
    const Eigen::Matrix<double, 11, 11> A = I - K * H;
    P_ = A * P_ * A.transpose() + K * R_eff * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

void ArmorEKF::updateAngularVelocity(double measured_w, double variance) {
    if (!initialized_ || !std::isfinite(measured_w) ||
        !std::isfinite(variance) || variance <= 0.0) {
        return;
    }

    Eigen::Matrix<double, 1, 11> H = Eigen::Matrix<double, 1, 11>::Zero();
    H(0, 7) = 1.0;

    const double innovation = measured_w - X_(7);
    const double S = P_(7, 7) + variance;
    if (!std::isfinite(S) || S <= 1e-12) return;

    Eigen::Matrix<double, 11, 1> K = P_.col(7) / S;
    // Angular velocity directly observes only w. Do not let its correction
    // rewrite static geometry through accumulated cross covariance.
    K(8) = 0.0;
    K(9) = 0.0;
    K(10) = 0.0;
    X_ += K * innovation;
    enforcePhysicalLimits();

    const Eigen::Matrix<double, 11, 11> I = Eigen::Matrix<double, 11, 11>::Identity();
    const Eigen::Matrix<double, 11, 11> A = I - K * H;
    P_ = A * P_ * A.transpose() + K * variance * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

std::array<double, 3> ArmorEKF::geometryVariances() const {
    if (!initialized_) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan, nan};
    }
    return {P_(8, 8), P_(9, 9), P_(10, 10)};
}

void ArmorEKF::setGeometryVariances(double var_r1,
                                    double var_r2,
                                    double var_h) {
    if (!initialized_) return;
    // GeometryMemory intentionally stores the three parameter variances only.
    // A soft-loss reinitialization starts a new motion covariance, so discard
    // stale geometry cross-covariances instead of mixing them across epochs.
    P_.block<3, 3>(8, 8).setZero();
    if (std::isfinite(var_r1) && var_r1 >= 0.0) P_(8, 8) = var_r1;
    if (std::isfinite(var_r2) && var_r2 >= 0.0) P_(9, 9) = var_r2;
    if (std::isfinite(var_h) && var_h >= 0.0) P_(10, 10) = var_h;
    P_ = 0.5 * (P_ + P_.transpose());
}

const char* trackerStateName(TrackerState state) {
    switch (state) {
        case TrackerState::LOST: return "LOST";
        case TrackerState::DETECTING: return "DETECTING";
        case TrackerState::TRACKING: return "TRACKING";
        case TrackerState::TEMP_LOST: return "TEMP_LOST";
    }
    return "UNKNOWN";
}

void RobustArmorTracker::configure(const RobustTrackerConfig& cfg) {
    cfg_ = cfg;
    ekf_.configure(cfg_);
    clear();
}

void RobustArmorTracker::resetPhaseObserver() {
    have_phase_yaw_ = false;
    last_phase_yaw_ = 0.0;
    phase_elapsed_ = 0.0;
    phase_w_valid_ = false;
    phase_w_filtered_ = 0.0;
    reversal_conflict_count_ = 0;
    reversal_conflict_sum_ = 0.0;
}

void RobustArmorTracker::loseTrackPreserveGeometry() {
    ekf_.invalidate();
    state_ = TrackerState::LOST;
    current_armor_id_ = -1;
    detect_count_ = 0;
    detect_misses_ = 0;
    lost_frames_ = 0;
    stable_association_frames_ = 0;
    resetPhaseObserver();
}

void RobustArmorTracker::clear() {
    loseTrackPreserveGeometry();
    geometry_memory_ = GeometryMemory{};
}

void RobustArmorTracker::reset(const ArmorState& state,
                               int physical_armor_id,
                               TrackerState tracker_state) {
    ekf_.reset(state);
    state_ = tracker_state;
    current_armor_id_ = std::clamp(physical_armor_id, 0, 3);
    detect_count_ = (tracker_state == TrackerState::TRACKING)
                        ? std::max(1, cfg_.detect_confirm_frames)
                        : 1;
    detect_misses_ = 0;
    lost_frames_ = 0;
    stable_association_frames_ = 0;
    if (geometry_memory_.valid) {
        double var_r1 = geometry_memory_.var_r1;
        double var_r2 = geometry_memory_.var_r2;
        double var_h = geometry_memory_.var_h;
        if (cfg_.geometry_reinit_covariance_floor_enabled) {
            const double radius_floor = std::max(
                0.0, cfg_.geometry_reinit_radius_variance_floor);
            const double height_floor = std::max(
                0.0, cfg_.geometry_reinit_height_variance_floor);
            var_r1 = std::max(var_r1, radius_floor);
            var_r2 = std::max(var_r2, radius_floor);
            var_h = std::max(var_h, height_floor);
        }
        ekf_.setGeometryVariances(
            var_r1, var_r2, var_h);
    }
    resetPhaseObserver();
}

bool RobustArmorTracker::geometryStateValid(const ArmorState& state) const {
    return std::isfinite(state.r1) && std::isfinite(state.r2) &&
           std::isfinite(state.h) &&
           state.r1 >= cfg_.state_min_radius &&
           state.r1 <= cfg_.state_max_radius &&
           state.r2 >= cfg_.state_min_radius &&
           state.r2 <= cfg_.state_max_radius &&
           std::abs(state.h) <= cfg_.state_max_abs_h;
}

void RobustArmorTracker::updateGeometryMemory() {
    if (state_ != TrackerState::TRACKING || !ekf_.initialized()) return;
    const ArmorState geometry = ekf_.state();
    const std::array<double, 3> variances = ekf_.geometryVariances();
    if (!geometryStateValid(geometry) ||
        !std::isfinite(variances[0]) || variances[0] < 0.0 ||
        !std::isfinite(variances[1]) || variances[1] < 0.0 ||
        !std::isfinite(variances[2]) || variances[2] < 0.0) {
        return;
    }
    geometry_memory_.r1 = geometry.r1;
    geometry_memory_.r2 = geometry.r2;
    geometry_memory_.h = geometry.h;
    geometry_memory_.var_r1 = variances[0];
    geometry_memory_.var_r2 = variances[1];
    geometry_memory_.var_h = variances[2];
    geometry_memory_.valid = true;
}

void RobustArmorTracker::populateGeometryDebug(TrackerResult& result) const {
    result.geometry_valid = geometry_memory_.valid;
}

bool RobustArmorTracker::validInput(const ArmorObservation& obs) const {
    if (!std::isfinite(obs.x) || !std::isfinite(obs.y) ||
        !std::isfinite(obs.z) || !std::isfinite(obs.yaw)) {
        return false;
    }
    const double range = std::sqrt(obs.x * obs.x + obs.y * obs.y + obs.z * obs.z);
    if (range < cfg_.input_min_range || range > cfg_.input_max_range) return false;
    if (std::abs(obs.z) > cfg_.input_max_abs_z) return false;
    return true;
}

ArmorState RobustArmorTracker::initializeStateFromMeasurement(
    const ArmorObservation& obs, int armor_id) const {
    const int id = std::clamp(armor_id, 0, 3);
    ArmorState s = cfg_.initial_filter;
    if (geometry_memory_.valid) {
        s.r1 = geometry_memory_.r1;
        s.r2 = geometry_memory_.r2;
        s.h = geometry_memory_.h;
    }
    const double r = (id % 2 == 0) ? s.r1 : s.r2;
    const double dz = (id % 2 == 0) ? 0.0 : s.h;

    s.yaw = wrapAngle(obs.yaw - id * kPi / 2.0);
    s.x = obs.x - r * std::sin(obs.yaw);
    s.y = obs.y + r * std::cos(obs.yaw);
    s.z = obs.z - dz;
    s.vx = 0.0;
    s.vy = 0.0;
    s.vz = 0.0;
    s.w = 0.0;
    return s;
}

RobustArmorTracker::PhaseUpdate RobustArmorTracker::observeRotationPhase(
    const ArmorObservation& obs, double dt) {
    PhaseUpdate out;
    if (!cfg_.rotation_phase_observer_enable) return out;

    if (have_phase_yaw_ && dt > 0.0 && std::isfinite(dt)) phase_elapsed_ += dt;

    if (!have_phase_yaw_) {
        have_phase_yaw_ = true;
        last_phase_yaw_ = obs.yaw;
        phase_elapsed_ = 0.0;
        return out;
    }

    const double elapsed = phase_elapsed_;
    phase_elapsed_ = 0.0;
    const double raw_delta = obs.yaw - last_phase_yaw_;
    last_phase_yaw_ = obs.yaw;

    if (!(elapsed > cfg_.rotation_phase_min_dt) ||
        elapsed > cfg_.rotation_phase_max_dt) {
        return out;
    }

    constexpr double period = kPi / 2.0;
    const double delta = std::remainder(raw_delta, period);
    const double instant_w = delta / elapsed;

    if (!std::isfinite(instant_w) ||
        std::abs(instant_w) > cfg_.rotation_phase_max_abs_omega ||
        std::abs(delta) > deg2rad(cfg_.rotation_phase_max_step_deg)) {
        return out;
    }

    if (!phase_w_valid_) {
        phase_w_filtered_ = instant_w;
        phase_w_valid_ = true;
    } else {
        const double alpha = std::clamp(cfg_.rotation_phase_alpha, 0.0, 1.0);
        phase_w_filtered_ = (1.0 - alpha) * phase_w_filtered_ + alpha * instant_w;
    }

    out.valid = true;
    out.delta = delta;
    out.instant_w = instant_w;
    out.filtered_w = phase_w_filtered_;

    if (!ekf_.initialized()) return out;

    const double ekf_w = ekf_.state().w;
    const double min_w = std::max(0.0, cfg_.rotation_reversal_min_abs_omega);
    const bool sign_conflict =
        std::abs(instant_w) >= min_w &&
        std::abs(ekf_w) >= min_w &&
        instant_w * ekf_w < 0.0;

    if (sign_conflict) {
        ++reversal_conflict_count_;
        reversal_conflict_sum_ += instant_w;
        out.pending_sign_conflict = true;

        if (reversal_conflict_count_ >= std::max(1, cfg_.rotation_reversal_confirm_frames)) {
            const double reversal_w =
                reversal_conflict_sum_ / static_cast<double>(reversal_conflict_count_);
            phase_w_filtered_ = reversal_w;
            out.filtered_w = reversal_w;
            out.reversal_confirmed = true;
            out.pending_sign_conflict = false;
            reversal_conflict_count_ = 0;
            reversal_conflict_sum_ = 0.0;
            ekf_.updateAngularVelocity(reversal_w, cfg_.rotation_reversal_w_variance);
        }
    } else {
        reversal_conflict_count_ = 0;
        reversal_conflict_sum_ = 0.0;
    }

    return out;
}

AssociationCandidate RobustArmorTracker::chooseAssociation(
    const Eigen::Matrix<double, 4, 1>& z,
    int forced_physical_armor_id,
    bool phase_conflict,
    bool* armor_switched,
    std::array<AssociationHypothesisDebug, 4>* debug_hypotheses) const {
    if (armor_switched) *armor_switched = false;

    auto passesGate = [this](const AssociationCandidate& c) {
        return c.numerically_valid &&
               c.nis <= cfg_.association_nis_gate &&
               c.position_error <= cfg_.association_max_position_error &&
               c.yaw_error <= deg2rad(cfg_.association_max_yaw_error_deg);
    };

    AssociationCandidate best;
    best.nis = std::numeric_limits<double>::infinity();
    AssociationCandidate current;
    bool have_current = false;

    const ArmorState predicted_state = ekf_.state();
    int visible_count = 0;
    const bool recovery = state_ == TrackerState::TEMP_LOST;
    auto yawScaleForCandidate = [this, recovery, phase_conflict](int id) {
        const bool candidate_switch =
            current_armor_id_ >= 0 && id != current_armor_id_;
        const bool candidate_topology_event =
            candidate_switch || recovery || phase_conflict;
        return candidate_topology_event
            ? cfg_.rotation_switch_yaw_r_scale : 1.0;
    };

    for (int id = 0; id < 4; ++id) {
        const ArmorObservation armor = ArmorModel::getArmor(
            predicted_state, id, 0.0);
        const double range = std::sqrt(armor.x * armor.x + armor.y * armor.y + armor.z * armor.z);
        const double facing = ArmorModel::facingScore(predicted_state, armor, 0.0);
        const bool range_pass =
            range >= cfg_.input_min_range && range <= cfg_.input_max_range;
        const bool visibility_pass =
            facing <= deg2rad(cfg_.armor_visible_angle_deg);
        const bool formal_eligible =
            range_pass &&
            (visibility_pass || id == current_armor_id_);
        if (!formal_eligible && debug_hypotheses == nullptr) continue;
        // Evaluate with the exact yaw covariance this candidate will use if it
        // is selected. Current normal-tracking continuity stays at scale 1.
        const AssociationCandidate c = ekf_.evaluateMeasurement(
            z, id, yawScaleForCandidate(id));

        if (debug_hypotheses != nullptr) {
            AssociationHypothesisDebug& debug = (*debug_hypotheses)[id];
            debug.armor_id = id;
            debug.predicted = armor;
            debug.facing_angle = facing;
            debug.range_pass = range_pass;
            debug.visibility_pass = visibility_pass;
            debug.measurement = c;
            debug.hypothetical_scaled_yaw_measurement =
                ekf_.evaluateMeasurement(
                    z, id, cfg_.rotation_switch_yaw_r_scale);
            const double sin_yaw = std::sin(armor.yaw);
            const double cos_yaw = std::cos(armor.yaw);
            debug.radial_residual =
                c.innovation(0) * sin_yaw - c.innovation(1) * cos_yaw;
            debug.tangential_residual =
                c.innovation(0) * cos_yaw + c.innovation(1) * sin_yaw;
            debug.nis_gate_pass =
                c.numerically_valid && c.nis <= cfg_.association_nis_gate;
            debug.position_gate_pass =
                c.numerically_valid &&
                c.position_error <= cfg_.association_max_position_error;
            debug.yaw_gate_pass =
                c.numerically_valid &&
                c.yaw_error <= deg2rad(cfg_.association_max_yaw_error_deg);
            debug.passes_all_measurement_gates = passesGate(c);
        }

        // The current armor is a continuity candidate. Visibility may filter
        // switch candidates, but it must not delete a current hypothesis that
        // still passes the unchanged measurement gates.
        if (!formal_eligible) continue;

        ++visible_count;
        if (id == current_armor_id_) {
            current = c;
            have_current = true;
        }
        if (c.numerically_valid && c.nis < best.nis) best = c;
    }

    // If geometry became temporarily inconsistent, do not hard-deadlock association.
    if (visible_count == 0) {
        for (int id = 0; id < 4; ++id) {
            const AssociationCandidate& c =
                debug_hypotheses != nullptr
                    ? (*debug_hypotheses)[id].measurement
                    : ekf_.evaluateMeasurement(
                          z, id, yawScaleForCandidate(id));
            if (id == current_armor_id_) {
                current = c;
                have_current = true;
            }
            if (c.numerically_valid && c.nis < best.nis) best = c;
        }
    }

    if (forced_physical_armor_id >= 0 && forced_physical_armor_id < 4) {
        return debug_hypotheses != nullptr
            ? (*debug_hypotheses)[forced_physical_armor_id].measurement
            : ekf_.evaluateMeasurement(
                  z, forced_physical_armor_id,
                  yawScaleForCandidate(forced_physical_armor_id));
    }

    if (!best.numerically_valid || !passesGate(best)) return best;

    if (current_armor_id_ >= 0 && best.armor_id != current_armor_id_ &&
        have_current && passesGate(current)) {
        if (best.nis + cfg_.association_switch_nis_margin >= current.nis) {
            best = current;
        }
    }

    if (armor_switched && current_armor_id_ >= 0 &&
        best.armor_id != current_armor_id_) {
        *armor_switched = true;
    }
    return best;
}

TrackerResult RobustArmorTracker::handleMiss(TrackerResult result) {
    result.updated = false;
    stable_association_frames_ = 0;

    switch (state_) {
        case TrackerState::LOST:
            break;
        case TrackerState::DETECTING:
            ++detect_misses_;
            detect_count_ = 0;
            if (detect_misses_ > cfg_.detect_max_misses) {
                const bool preserve = geometry_memory_.valid;
                loseTrackPreserveGeometry();
                result.geometry_preserved = preserve;
            }
            break;
        case TrackerState::TRACKING:
            state_ = TrackerState::TEMP_LOST;
            lost_frames_ = 1;
            break;
        case TrackerState::TEMP_LOST:
            ++lost_frames_;
            if (lost_frames_ > cfg_.temp_lost_max_frames) {
                const bool preserve = geometry_memory_.valid;
                loseTrackPreserveGeometry();
                result.geometry_preserved = preserve;
            }
            break;
    }

    result.state = state_;
    result.lost_frames = lost_frames_;
    result.detect_count = detect_count_;
    populateGeometryDebug(result);
    return result;
}

TrackerResult RobustArmorTracker::process(
    const std::optional<ArmorObservation>& measurement,
    double dt,
    int forced_physical_armor_id) {
    TrackerResult result;
    result.tracker_state_before = state_;

    if (ekf_.initialized() && dt > 0.0) ekf_.predict(dt);

    if (!measurement.has_value() || !validInput(*measurement)) {
        if (have_phase_yaw_ && dt > 0.0 && std::isfinite(dt)) phase_elapsed_ += dt;
        result.measurement_valid = measurement.has_value() && validInput(*measurement);
        return handleMiss(result);
    }

    result.measurement_valid = true;
    const Eigen::Matrix<double, 4, 1> z = measurement->toVector();
    result.measurement_yaw = z(3);

    if (state_ == TrackerState::LOST || !ekf_.initialized()) {
        const int init_id =
            (forced_physical_armor_id >= 0 && forced_physical_armor_id < 4)
                ? forced_physical_armor_id : 0;
        const bool restore_geometry = geometry_memory_.valid;
        reset(initializeStateFromMeasurement(*measurement, init_id),
              init_id,
              TrackerState::DETECTING);
        have_phase_yaw_ = true;
        last_phase_yaw_ = measurement->yaw;
        phase_elapsed_ = 0.0;

        result.state = state_;
        result.matched_id = init_id;
        result.initialized_this_frame = true;
        result.detect_count = detect_count_;
        result.geometry_preserved = restore_geometry;
        populateGeometryDebug(result);
        return result;
    }

    const PhaseUpdate phase = observeRotationPhase(*measurement, dt);
    result.phase_observer_valid = phase.valid;
    result.phase_delta = phase.delta;
    result.phase_w_instant = phase.instant_w;
    result.phase_w_filtered = phase.filtered_w;
    result.direction_reversal = phase.reversal_confirmed;
    result.pending_sign_conflict = phase.pending_sign_conflict;

    bool armor_switched = false;
    AssociationCandidate chosen =
        chooseAssociation(z, forced_physical_armor_id,
                          phase.pending_sign_conflict, &armor_switched,
                          cfg_.association_debug_enable
                              ? &result.association_hypotheses : nullptr);

    result.current_armor_id = current_armor_id_;
    result.best_id = chosen.armor_id;
    result.candidate_is_switch =
        current_armor_id_ >= 0 && chosen.armor_id >= 0 &&
        chosen.armor_id != current_armor_id_;
    result.temp_lost_recovery =
        result.tracker_state_before == TrackerState::TEMP_LOST;
    result.topology_event =
        result.candidate_is_switch || result.temp_lost_recovery ||
        phase.pending_sign_conflict;
    result.predicted_yaw = chosen.predicted(3);
    result.yaw_innovation = chosen.innovation(3);
    if (cfg_.association_debug_enable && chosen.armor_id >= 0) {
        const AssociationCandidate hypothetical = ekf_.evaluateMeasurement(
            z, chosen.armor_id, cfg_.rotation_switch_yaw_r_scale);
        result.hypothetical_scaled_nis = hypothetical.nis;
        result.hypothetical_scaled_nis_contribution =
            hypothetical.nis_contribution;
    }

    result.nis = chosen.nis;
    result.position_error = chosen.position_error;
    result.yaw_error = chosen.yaw_error;

    const bool association_ok =
        chosen.numerically_valid &&
        chosen.nis <= cfg_.association_nis_gate &&
        chosen.position_error <= cfg_.association_max_position_error &&
        chosen.yaw_error <= deg2rad(cfg_.association_max_yaw_error_deg) &&
        chosen.armor_id >= 0;

    if (!association_ok) {
        if (phase.valid && !phase.pending_sign_conflict && !phase.reversal_confirmed) {
            ekf_.updateAngularVelocity(phase.filtered_w, cfg_.rotation_phase_w_variance);
            result.phase_w_applied = true;
        } else if (phase.reversal_confirmed) {
            result.phase_w_applied = true;
        }
        return handleMiss(result);
    }

    const TrackerState before = state_;
    const bool recovered = before == TrackerState::TEMP_LOST;
    const bool stable_geometry_association =
        before == TrackerState::TRACKING &&
        !armor_switched &&
        !phase.reversal_confirmed &&
        !phase.pending_sign_conflict &&
        !recovered;
    if (stable_geometry_association) {
        ++stable_association_frames_;
    } else {
        stable_association_frames_ = 0;
    }
    const bool geometry_update_allowed =
        stable_geometry_association &&
        stable_association_frames_ >=
            std::max(1, cfg_.geometry_stable_frames_before_update);
    // AssociationCandidate owns the future-update measurement covariance.
    // Do not derive topology or yaw scale again here.
    ekf_.update(z, chosen.armor_id, chosen.yaw_variance_scale,
                geometry_update_allowed);

    if (phase.valid) {
        if (phase.reversal_confirmed) {
            ekf_.updateAngularVelocity(
                phase.filtered_w, cfg_.rotation_reversal_w_variance);
            result.phase_w_applied = true;
        } else if (!phase.pending_sign_conflict) {
            ekf_.updateAngularVelocity(
                phase.filtered_w, cfg_.rotation_phase_w_variance);
            result.phase_w_applied = true;
        }
    }

    current_armor_id_ = chosen.armor_id;
    lost_frames_ = 0;
    detect_misses_ = 0;

    if (state_ == TrackerState::DETECTING) {
        ++detect_count_;
        if (detect_count_ >= std::max(1, cfg_.detect_confirm_frames)) {
            state_ = TrackerState::TRACKING;
        }
    } else {
        state_ = TrackerState::TRACKING;
    }

    if (geometry_update_allowed) updateGeometryMemory();

    result.state = state_;
    result.matched_id = chosen.armor_id;
    result.updated = true;
    result.recovered = recovered;
    result.armor_switched = armor_switched;
    result.geometry_update_allowed = geometry_update_allowed;
    result.lost_frames = lost_frames_;
    result.detect_count = detect_count_;
    populateGeometryDebug(result);
    return result;
}

}  // namespace rm_ekf
