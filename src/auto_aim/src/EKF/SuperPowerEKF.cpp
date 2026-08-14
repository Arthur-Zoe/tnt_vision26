#include "EKF/SuperPowerEKF.h"

#include <numeric>
#include <utility>

namespace sp_ekf {

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> x_add)
    : x(x0),
      P(P0),
      I_(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())),
      x_add_(std::move(x_add)) {
    data["residual_yaw"] = 0.0;
    data["residual_pitch"] = 0.0;
    data["residual_distance"] = 0.0;
    data["residual_angle"] = 0.0;
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd& F,
                                               const Eigen::MatrixXd& Q) {
    return predict(F, Q, [&](const Eigen::VectorXd& value) {
        return F * value;
    });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
    const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f) {
    P = F * P * F.transpose() + Q;
    x = f(x);
    return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
    const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> z_subtract) {
    return update(z, H, R,
                  [&](const Eigen::VectorXd& value) { return H * value; },
                  std::move(z_subtract));
}

Eigen::VectorXd ExtendedKalmanFilter::update(
    const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> z_subtract,
    const std::vector<int>& frozen_state_indices) {
    const Eigen::VectorXd x_prior = x;
    Eigen::MatrixXd K =
        P * H.transpose() * (H * P * H.transpose() + R).inverse();

    // Project-specific vertical stabilization: on an armor-topology switch,
    // Target can freeze selected state rows in the correction.  The normal
    // path passes an empty list and remains identical to SuperPower.
    for (const int index : frozen_state_indices) {
        if (index >= 0 && index < K.rows()) K.row(index).setZero();
    }

    // Joseph-form covariance update, matching SuperPower.
    P = (I_ - K * H) * P * (I_ - K * H).transpose() + K * R * K.transpose();
    x = x_add_(x, K * z_subtract(z, h(x)));

    // Keep SuperPower's post-update health statistics exactly in the estimator
    // path. They are diagnostic/reset signals, not an extra association gate.
    const Eigen::VectorXd residual = z_subtract(z, h(x));
    const Eigen::MatrixXd S = H * P * H.transpose() + R;
    const double nis = residual.transpose() * S.inverse() * residual;
    const double nees =
        (x - x_prior).transpose() * P.inverse() * (x - x_prior);

    constexpr double nis_threshold = 0.711;
    constexpr double nees_threshold = 0.711;

    if (nis > nis_threshold) {
        ++nis_count_;
        data["nis_fail"] = 1.0;
    }
    if (nees > nees_threshold) {
        ++nees_count_;
        data["nees_fail"] = 1.0;
    }
    ++total_count_;
    last_nis = nis;

    recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);
    if (recent_nis_failures.size() > window_size) {
        recent_nis_failures.pop_front();
    }

    const int recent_failures =
        std::accumulate(recent_nis_failures.begin(),
                        recent_nis_failures.end(), 0);
    const double recent_rate = static_cast<double>(recent_failures) /
        static_cast<double>(recent_nis_failures.size());

    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
    data["nis"] = nis;
    data["nees"] = nees;
    data["recent_nis_failures"] = recent_rate;

    return x;
}

}  // namespace sp_ekf
