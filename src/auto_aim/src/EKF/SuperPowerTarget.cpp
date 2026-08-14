#include "EKF/SuperPowerTarget.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sp_ekf {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

Target::Target(const ArmorObservation& armor,
               double radius,
               int armor_num,
               const Eigen::VectorXd& P0_diag)
    : armor_num_(armor_num) {
    const double center_x = armor.xyz[0] + radius * std::cos(armor.angle);
    const double center_y = armor.xyz[1] + radius * std::sin(armor.angle);
    const double center_z = armor.xyz[2];

    // SuperPower state order: x vx y vy z vz a w r l h
    Eigen::VectorXd x0(11);
    x0 << center_x, 0.0,
          center_y, 0.0,
          center_z, 0.0,
          armor.angle, 0.0,
          radius, 0.0, 0.0;

    const Eigen::MatrixXd P0 = P0_diag.asDiagonal();
    auto x_add = [](const Eigen::VectorXd& a,
                    const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a + b;
        while (c[6] > kPi) c[6] -= 2.0 * kPi;
        while (c[6] <= -kPi) c[6] += 2.0 * kPi;
        return c;
    };

    ekf_ = ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(double dt) {
    Eigen::MatrixXd F(11, 11);
    F << 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,  1, dt,0, 0, 0, 0, 0, 0, 0,
         0, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0,  0, 0, 1, dt,0, 0, 0, 0, 0,
         0, 0,  0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0,  0, 0, 0, 0, 1, dt,0, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1;

    // Normal four-armor vehicle values from SuperPower Target::predict().
    constexpr double v1 = 100.0;
    constexpr double v2 = 400.0;
    const double a = dt * dt * dt * dt / 4.0;
    const double b = dt * dt * dt / 2.0;
    const double c = dt * dt;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0,0)=a*v1; Q(0,1)=b*v1; Q(1,0)=b*v1; Q(1,1)=c*v1;
    Q(2,2)=a*v1; Q(2,3)=b*v1; Q(3,2)=b*v1; Q(3,3)=c*v1;
    Q(4,4)=a*v1; Q(4,5)=b*v1; Q(5,4)=b*v1; Q(5,5)=c*v1;
    Q(6,6)=a*v2; Q(6,7)=b*v2; Q(7,6)=b*v2; Q(7,7)=c*v2;

    auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd prior = F * x;
        prior[6] = limitRad(prior[6]);
        return prior;
    };
    ekf_.predict(F, Q, f);
}

TargetUpdateDebug Target::update(const ArmorObservation& armor) {
    TargetUpdateDebug debug;
    const std::vector<Eigen::Vector4d> xyza_list = armorXyzaList();

    std::vector<std::pair<Eigen::Vector4d, int>> candidates;
    candidates.reserve(armor_num_);
    for (int i = 0; i < armor_num_; ++i) {
        candidates.push_back({xyza_list[static_cast<std::size_t>(i)], i});
    }

    std::sort(
        candidates.begin(), candidates.end(),
        [](const std::pair<Eigen::Vector4d, int>& lhs,
           const std::pair<Eigen::Vector4d, int>& rhs) {
            return lhs.first.head<3>().norm() < rhs.first.head<3>().norm();
        });

    int id = 0;
    double min_angle_error = std::numeric_limits<double>::infinity();
    const Eigen::Vector3d armor_ypd = xyz2ypd(armor.xyz);
    const int inspect_count = std::min(3, armor_num_);
    for (int i = 0; i < inspect_count; ++i) {
        const Eigen::Vector4d& xyza = candidates[static_cast<std::size_t>(i)].first;
        const Eigen::Vector3d ypd = xyz2ypd(xyza.head<3>());
        const double angle_error =
            std::abs(limitRad(armor.angle - xyza[3])) +
            std::abs(limitRad(armor_ypd[0] - ypd[0]));
        if (std::abs(angle_error) < std::abs(min_angle_error)) {
            id = candidates[static_cast<std::size_t>(i)].second;
            min_angle_error = angle_error;
        }
    }

    debug.matched_id = id;
    debug.armor_switched = (id != last_id_);
    debug.predicted_xyza = xyza_list[static_cast<std::size_t>(id)];
    debug.position_error =
        (armor.xyz - debug.predicted_xyza.head<3>()).norm();
    debug.angle_error = std::abs(limitRad(armor.angle - debug.predicted_xyza[3]));

    is_switch_ = debug.armor_switched;
    if (is_switch_) ++switch_count_;
    last_id_ = id;
    ++update_count_;

    updateYpda(armor, id);
    debug.nis = ekf_.last_nis;
    return debug;
}

void Target::updateYpda(const ArmorObservation& armor, int id) {
    const Eigen::MatrixXd H = hJacobian(ekf_.x, id);

    const double center_yaw = std::atan2(armor.xyz[1], armor.xyz[0]);
    const double delta_angle = limitRad(armor.angle - center_yaw);
    const Eigen::Vector3d ypd = xyz2ypd(armor.xyz);

    Eigen::VectorXd R_diag(4);
    R_diag << 4e-3,
              4e-3,
              std::log(std::abs(delta_angle) + 1.0) + 1.0,
              std::log(std::abs(ypd[2]) + 1.0) / 200.0 + 9e-2;
    const Eigen::MatrixXd R = R_diag.asDiagonal();

    auto h = [&](const Eigen::VectorXd& x) -> Eigen::Vector4d {
        const Eigen::Vector3d xyz = armorXyz(x, id);
        const Eigen::Vector3d local_ypd = xyz2ypd(xyz);
        const double angle = limitRad(
            x[6] + id * 2.0 * kPi / static_cast<double>(armor_num_));
        return {local_ypd[0], local_ypd[1], local_ypd[2], angle};
    };

    auto z_subtract = [](const Eigen::VectorXd& a,
                         const Eigen::VectorXd& b) -> Eigen::VectorXd {
        Eigen::VectorXd c = a - b;
        while (c[0] > kPi) c[0] -= 2.0 * kPi;
        while (c[0] <= -kPi) c[0] += 2.0 * kPi;
        while (c[1] > kPi) c[1] -= 2.0 * kPi;
        while (c[1] <= -kPi) c[1] += 2.0 * kPi;
        while (c[3] > kPi) c[3] -= 2.0 * kPi;
        while (c[3] <= -kPi) c[3] += 2.0 * kPi;
        return c;
    };

    Eigen::VectorXd z(4);
    z << ypd[0], ypd[1], ypd[2], armor.angle;
    // Vertical observability fix for the single-armor input used by this
    // project.  For a four-armor target, E0/E2 measure the base layer z, while
    // E1/E3 measure z+h.  While only an odd armor is visible, z and h are not
    // separately observable from that one measurement.  Allowing the normal
    // Kalman gain to update both makes the whole target drift vertically after
    // a switch.
    //
    // Keep E0/E2 as the anchor for center z.  During every E1/E3 update (not
    // just the first switch frame), freeze center z and vz so the vertical
    // innovation is absorbed by h.  This leaves SP's XY/yaw/w/r/l update path
    // unchanged and makes the two armor layers explicit:
    //   E0/E2: z_armour = center_z
    //   E1/E3: z_armour = center_z + h
    const bool odd_height_layer =
        armor_num_ == 4 && (id == 1 || id == 3);
    const std::vector<int> frozen_states = odd_height_layer
        ? std::vector<int>{4, 5}
        : std::vector<int>{};
    ekf_.update(z, H, R, h, z_subtract, frozen_states);
}

Eigen::VectorXd Target::ekfX() const { return ekf_.x; }

const ExtendedKalmanFilter& Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armorXyzaList() const {
    std::vector<Eigen::Vector4d> result;
    result.reserve(static_cast<std::size_t>(armor_num_));
    for (int i = 0; i < armor_num_; ++i) {
        const double angle = limitRad(
            ekf_.x[6] + i * 2.0 * kPi / static_cast<double>(armor_num_));
        const Eigen::Vector3d xyz = armorXyz(ekf_.x, i);
        result.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return result;
}

bool Target::diverged() const {
    const bool r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    const bool l_ok = (ekf_.x[8] + ekf_.x[9]) > 0.05 &&
                      (ekf_.x[8] + ekf_.x[9]) < 0.5;
    return !(r_ok && l_ok);
}

bool Target::converged() {
    if (update_count_ > 3 && !diverged()) {
        is_converged_ = true;
    }
    return is_converged_;
}

Eigen::Vector3d Target::armorXyz(const Eigen::VectorXd& x, int id) const {
    const double angle = limitRad(
        x[6] + id * 2.0 * kPi / static_cast<double>(armor_num_));
    const bool use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
    const double r = use_l_h ? x[8] + x[9] : x[8];
    const double armor_x = x[0] - r * std::cos(angle);
    const double armor_y = x[2] - r * std::sin(angle);
    const double armor_z = use_l_h ? x[4] + x[10] : x[4];
    return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::hJacobian(const Eigen::VectorXd& x, int id) const {
    const double angle = limitRad(
        x[6] + id * 2.0 * kPi / static_cast<double>(armor_num_));
    const bool use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
    const double r = use_l_h ? x[8] + x[9] : x[8];

    const double dx_da = r * std::sin(angle);
    const double dy_da = -r * std::cos(angle);
    const double dx_dr = -std::cos(angle);
    const double dy_dr = -std::sin(angle);
    const double dx_dl = use_l_h ? -std::cos(angle) : 0.0;
    const double dy_dl = use_l_h ? -std::sin(angle) : 0.0;
    const double dz_dh = use_l_h ? 1.0 : 0.0;

    Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
    H_armor_xyza(0,0)=1.0; H_armor_xyza(0,6)=dx_da;
    H_armor_xyza(0,8)=dx_dr; H_armor_xyza(0,9)=dx_dl;
    H_armor_xyza(1,2)=1.0; H_armor_xyza(1,6)=dy_da;
    H_armor_xyza(1,8)=dy_dr; H_armor_xyza(1,9)=dy_dl;
    H_armor_xyza(2,4)=1.0; H_armor_xyza(2,10)=dz_dh;
    H_armor_xyza(3,6)=1.0;

    const Eigen::Vector3d armor_xyz = armorXyz(x, id);
    const Eigen::Matrix3d H_armor_ypd = xyz2ypdJacobian(armor_xyz);

    Eigen::Matrix4d H_armor_ypda = Eigen::Matrix4d::Zero();
    H_armor_ypda.block<3,3>(0,0) = H_armor_ypd;
    H_armor_ypda(3,3) = 1.0;
    return H_armor_ypda * H_armor_xyza;
}

double Target::limitRad(double angle) {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle <= -kPi) angle += 2.0 * kPi;
    return angle;
}

Eigen::Vector3d Target::xyz2ypd(const Eigen::Vector3d& xyz) {
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = xyz[2];
    const double yaw = std::atan2(y, x);
    const double horizontal = std::sqrt(x * x + y * y);
    const double pitch = std::atan2(z, horizontal);
    const double distance = std::sqrt(x * x + y * y + z * z);
    return {yaw, pitch, distance};
}

Eigen::Matrix3d Target::xyz2ypdJacobian(const Eigen::Vector3d& xyz) {
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = xyz[2];

    // Same analytic Jacobian as SuperPower.  The tiny floor prevents a NaN
    // only at the coordinate-system singularity (target exactly on z-axis).
    const double xy2 = std::max(x * x + y * y, 1e-12);
    const double xy = std::sqrt(xy2);
    const double d2 = std::max(xy2 + z * z, 1e-12);
    const double d = std::sqrt(d2);

    const double dyaw_dx = -y / xy2;
    const double dyaw_dy = x / xy2;

    const double dpitch_dx = -(x * z) / (d2 * xy);
    const double dpitch_dy = -(y * z) / (d2 * xy);
    const double dpitch_dz = xy / d2;

    const double ddistance_dx = x / d;
    const double ddistance_dy = y / d;
    const double ddistance_dz = z / d;

    Eigen::Matrix3d J;
    J << dyaw_dx,      dyaw_dy,      0.0,
         dpitch_dx,    dpitch_dy,    dpitch_dz,
         ddistance_dx, ddistance_dy, ddistance_dz;
    return J;
}

}  // namespace sp_ekf
