#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "EKF/SuperPowerEKF.h"

// Algorithm-space types.  All values are SuperPower convention:
// meter, second, radian; armor angle is the outward-normal yaw used by SP.
namespace sp_ekf {

struct ArmorObservation {
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
    double angle = 0.0;
};

struct TargetUpdateDebug {
    int matched_id = -1;
    bool armor_switched = false;
    Eigen::Vector4d predicted_xyza = Eigen::Vector4d::Zero();
    double position_error = -1.0;
    double angle_error = -1.0;
    double nis = -1.0;
};

class Target {
public:
    Target() = default;
    Target(const ArmorObservation& armor,
           double radius,
           int armor_num,
           const Eigen::VectorXd& P0_diag);

    void predict(double dt);
    TargetUpdateDebug update(const ArmorObservation& armor);

    Eigen::VectorXd ekfX() const;
    const ExtendedKalmanFilter& ekf() const;
    std::vector<Eigen::Vector4d> armorXyzaList() const;

    bool diverged() const;
    bool converged();
    int lastId() const { return last_id_; }
    int armorNum() const { return armor_num_; }

private:
    int armor_num_ = 4;
    int switch_count_ = 0;
    int update_count_ = 0;
    int last_id_ = 0;
    bool is_switch_ = false;
    bool is_converged_ = false;

    ExtendedKalmanFilter ekf_;

    void updateYpda(const ArmorObservation& armor, int id);
    Eigen::Vector3d armorXyz(const Eigen::VectorXd& x, int id) const;
    Eigen::MatrixXd hJacobian(const Eigen::VectorXd& x, int id) const;

    static double limitRad(double angle);
    static Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);
    static Eigen::Matrix3d xyz2ypdJacobian(const Eigen::Vector3d& xyz);
};

}  // namespace sp_ekf
