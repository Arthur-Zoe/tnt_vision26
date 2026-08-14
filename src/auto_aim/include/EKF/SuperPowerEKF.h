#pragma once

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

// Ported from TongjiSuperPower/sp_vision_25
// Source commit: ce3e1ce05ea57bef9813bba0c450ff158388f0a2
// Upstream file: tools/extended_kalman_filter.{hpp,cpp}
// License: MIT (see SUPERPOWER_NOTICE.md)
namespace sp_ekf {

class ExtendedKalmanFilter {
public:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;

    ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(
        const Eigen::VectorXd& x0,
        const Eigen::MatrixXd& P0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                      const Eigen::VectorXd&)> x_add =
            [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                return a + b;
            });

    Eigen::VectorXd predict(const Eigen::MatrixXd& F,
                            const Eigen::MatrixXd& Q);

    Eigen::VectorXd predict(
        const Eigen::MatrixXd& F,
        const Eigen::MatrixXd& Q,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f);

    Eigen::VectorXd update(
        const Eigen::VectorXd& z,
        const Eigen::MatrixXd& H,
        const Eigen::MatrixXd& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                      const Eigen::VectorXd&)> z_subtract =
            [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                return a - b;
            });

    Eigen::VectorXd update(
        const Eigen::VectorXd& z,
        const Eigen::MatrixXd& H,
        const Eigen::MatrixXd& R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                      const Eigen::VectorXd&)> z_subtract =
            [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                return a - b;
            },
        const std::vector<int>& frozen_state_indices = {});

    std::map<std::string, double> data;
    std::deque<int> recent_nis_failures{0};
    std::size_t window_size = 100;
    double last_nis = 0.0;

private:
    Eigen::MatrixXd I_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> x_add_;

    int nees_count_ = 0;
    int nis_count_ = 0;
    int total_count_ = 0;
};

}  // namespace sp_ekf
