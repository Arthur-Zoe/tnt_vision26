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

    readIfExists(n, "q_pos", cfg.q_pos);
    readIfExists(n, "q_vel", cfg.q_vel);
    readIfExists(n, "q_z", cfg.q_z);
    readIfExists(n, "q_vz", cfg.q_vz);
    readIfExists(n, "q_yaw", cfg.q_yaw);
    readIfExists(n, "q_w", cfg.q_w);
    readIfExists(n, "q_radius", cfg.q_radius);
    readIfExists(n, "q_h", cfg.q_h);

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

    return cfg;
}

ArmorObservation ArmorModel::getArmor(const ArmorState& s, int armor_id, double predict_time) {
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

std::vector<ArmorObservation> ArmorModel::getArmors(const ArmorState& s, double predict_time) {
    std::vector<ArmorObservation> armors;
    armors.reserve(4);
    for (int i = 0; i < 4; ++i) armors.push_back(getArmor(s, i, predict_time));
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
    Q_.setZero();
    Q_(0, 0) = cfg.q_pos;     Q_(1, 1) = cfg.q_vel;
    Q_(2, 2) = cfg.q_pos;     Q_(3, 3) = cfg.q_vel;
    Q_(4, 4) = cfg.q_z;       Q_(5, 5) = cfg.q_vz;
    Q_(6, 6) = cfg.q_yaw;     Q_(7, 7) = cfg.q_w;
    Q_(8, 8) = cfg.q_radius;  Q_(9, 9) = cfg.q_radius;
    Q_(10, 10) = cfg.q_h;

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

    X_ = F * X_;
    enforcePhysicalLimits();
    P_ = F * P_ * F.transpose() + Q_;
    P_ = 0.5 * (P_ + P_.transpose());
}

AssociationCandidate ArmorEKF::evaluateMeasurement(
    const Eigen::Matrix<double, 4, 1>& z, int armor_id) const {
    AssociationCandidate c;
    c.armor_id = ((armor_id % 4) + 4) % 4;
    if (!initialized_) return c;

    c.predicted = ArmorModel::measurementFunction(X_, c.armor_id);
    const Eigen::Matrix<double, 4, 11> H = ArmorModel::measurementJacobian(X_, c.armor_id);

    c.innovation = z - c.predicted;
    c.innovation(3) = wrapAngle(c.innovation(3));
    c.position_error = c.innovation.head<3>().norm();
    c.yaw_error = std::abs(c.innovation(3));

    const Eigen::Matrix<double, 4, 4> S = H * P_ * H.transpose() + R_;
    const auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success) return c;

    const Eigen::Matrix<double, 4, 1> solved = ldlt.solve(c.innovation);
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) return c;

    c.nis = c.innovation.dot(solved);
    c.numerically_valid = std::isfinite(c.nis) && c.nis >= 0.0;
    return c;
}

void ArmorEKF::update(const Eigen::Matrix<double, 4, 1>& z,
                      int armor_id,
                      double yaw_variance_scale) {
    if (!initialized_) return;

    const int id = ((armor_id % 4) + 4) % 4;
    const Eigen::Matrix<double, 4, 1> h = ArmorModel::measurementFunction(X_, id);
    const Eigen::Matrix<double, 4, 11> H = ArmorModel::measurementJacobian(X_, id);

    Eigen::Matrix<double, 4, 1> innovation = z - h;
    innovation(3) = wrapAngle(innovation(3));

    Eigen::Matrix<double, 4, 4> R_eff = R_;
    R_eff(3, 3) *= std::max(1.0, yaw_variance_scale);
    const Eigen::Matrix<double, 4, 4> S = H * P_ * H.transpose() + R_eff;
    const auto ldlt = S.ldlt();
    if (ldlt.info() != Eigen::Success) return;

    const Eigen::Matrix<double, 11, 4> PHt = P_ * H.transpose();
    const Eigen::Matrix<double, 4, 11> solved = ldlt.solve(PHt.transpose());
    if (ldlt.info() != Eigen::Success || !solved.allFinite()) return;
    const Eigen::Matrix<double, 11, 4> K = solved.transpose();

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

    const Eigen::Matrix<double, 11, 1> K = P_.col(7) / S;
    X_ += K * innovation;
    enforcePhysicalLimits();

    const Eigen::Matrix<double, 11, 11> I = Eigen::Matrix<double, 11, 11>::Identity();
    const Eigen::Matrix<double, 11, 11> A = I - K * H;
    P_ = A * P_ * A.transpose() + K * variance * K.transpose();
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

void RobustArmorTracker::clear() {
    ekf_.invalidate();
    state_ = TrackerState::LOST;
    current_armor_id_ = -1;
    detect_count_ = 0;
    detect_misses_ = 0;
    lost_frames_ = 0;
    resetPhaseObserver();
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
    resetPhaseObserver();
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
    bool* armor_switched) const {
    if (armor_switched) *armor_switched = false;

    auto passesGate = [this](const AssociationCandidate& c) {
        return c.numerically_valid &&
               c.nis <= cfg_.association_nis_gate &&
               c.position_error <= cfg_.association_max_position_error &&
               c.yaw_error <= deg2rad(cfg_.association_max_yaw_error_deg);
    };

    if (forced_physical_armor_id >= 0 && forced_physical_armor_id < 4) {
        return ekf_.evaluateMeasurement(z, forced_physical_armor_id);
    }

    AssociationCandidate best;
    best.nis = std::numeric_limits<double>::infinity();
    AssociationCandidate current;
    bool have_current = false;

    const ArmorState predicted_state = ekf_.state();
    int visible_count = 0;

    for (int id = 0; id < 4; ++id) {
        const ArmorObservation armor = ArmorModel::getArmor(predicted_state, id, 0.0);
        const double range = std::sqrt(armor.x * armor.x + armor.y * armor.y + armor.z * armor.z);
        if (range < cfg_.input_min_range || range > cfg_.input_max_range) continue;

        const double facing = ArmorModel::facingScore(predicted_state, armor, 0.0);
        if (facing > deg2rad(cfg_.armor_visible_angle_deg)) continue;

        ++visible_count;
        const AssociationCandidate c = ekf_.evaluateMeasurement(z, id);
        if (id == current_armor_id_) {
            current = c;
            have_current = true;
        }
        if (c.numerically_valid && c.nis < best.nis) best = c;
    }

    // If geometry became temporarily inconsistent, do not hard-deadlock association.
    if (visible_count == 0) {
        for (int id = 0; id < 4; ++id) {
            const AssociationCandidate c = ekf_.evaluateMeasurement(z, id);
            if (id == current_armor_id_) {
                current = c;
                have_current = true;
            }
            if (c.numerically_valid && c.nis < best.nis) best = c;
        }
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

    switch (state_) {
        case TrackerState::LOST:
            break;
        case TrackerState::DETECTING:
            ++detect_misses_;
            detect_count_ = 0;
            if (detect_misses_ > cfg_.detect_max_misses) clear();
            break;
        case TrackerState::TRACKING:
            state_ = TrackerState::TEMP_LOST;
            lost_frames_ = 1;
            break;
        case TrackerState::TEMP_LOST:
            ++lost_frames_;
            if (lost_frames_ > cfg_.temp_lost_max_frames) clear();
            break;
    }

    result.state = state_;
    result.lost_frames = lost_frames_;
    result.detect_count = detect_count_;
    return result;
}

TrackerResult RobustArmorTracker::process(
    const std::optional<ArmorObservation>& measurement,
    double dt,
    int forced_physical_armor_id) {
    TrackerResult result;

    if (ekf_.initialized() && dt > 0.0) ekf_.predict(dt);

    if (!measurement.has_value() || !validInput(*measurement)) {
        if (have_phase_yaw_ && dt > 0.0 && std::isfinite(dt)) phase_elapsed_ += dt;
        result.measurement_valid = measurement.has_value() && validInput(*measurement);
        return handleMiss(result);
    }

    result.measurement_valid = true;
    const Eigen::Matrix<double, 4, 1> z = measurement->toVector();

    if (state_ == TrackerState::LOST || !ekf_.initialized()) {
        const int init_id =
            (forced_physical_armor_id >= 0 && forced_physical_armor_id < 4)
                ? forced_physical_armor_id : 0;
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
        return result;
    }

    const PhaseUpdate phase = observeRotationPhase(*measurement, dt);
    result.phase_observer_valid = phase.valid;
    result.phase_delta = phase.delta;
    result.phase_w_instant = phase.instant_w;
    result.phase_w_filtered = phase.filtered_w;
    result.direction_reversal = phase.reversal_confirmed;

    bool armor_switched = false;
    AssociationCandidate chosen =
        chooseAssociation(z, forced_physical_armor_id, &armor_switched);

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
    const bool topology_event =
        armor_switched ||
        before == TrackerState::TEMP_LOST ||
        phase.pending_sign_conflict;
    const double yaw_r_scale =
        topology_event ? cfg_.rotation_switch_yaw_r_scale : 1.0;

    ekf_.update(z, chosen.armor_id, yaw_r_scale);

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

    result.state = state_;
    result.matched_id = chosen.armor_id;
    result.updated = true;
    result.recovered = (before == TrackerState::TEMP_LOST);
    result.armor_switched = armor_switched;
    result.lost_frames = lost_frames_;
    result.detect_count = detect_count_;
    return result;
}

}  // namespace rm_ekf
