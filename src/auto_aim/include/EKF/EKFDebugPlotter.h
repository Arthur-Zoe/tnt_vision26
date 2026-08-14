#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <string>

struct EKFDebugPlotConfig {
    bool enabled = false;
    bool csv_enabled = false;
    bool save_latest_png = false;
    double history_seconds = 10.0;
    int width = 1280;
    int panel_height = 120;
    std::string csv_path = "/tmp/ekf_debug_curves.csv";
    std::string screenshot_path = "/tmp/ekf_debug_curves.png";

    double r1_ref = 0.24;
    double r2_ref = 0.31;
    double h_ref = 0.09;
    double r1_tolerance = 0.03;
    double r2_tolerance = 0.03;
    double h_tolerance = 0.02;
    double nis_gate = 13.28;

    static EKFDebugPlotConfig fromYaml(const YAML::Node& root);
};

struct EKFDebugPlotSample {
    std::uint64_t frame_id = 0;
    double timestamp_s = 0.0;

    double measurement_x = std::numeric_limits<double>::quiet_NaN();
    double measurement_y = std::numeric_limits<double>::quiet_NaN();
    double measurement_z = std::numeric_limits<double>::quiet_NaN();
    double measurement_yaw = std::numeric_limits<double>::quiet_NaN();
    int matched_id = -1;

    double pre_pred_x = std::numeric_limits<double>::quiet_NaN();
    double pre_pred_y = std::numeric_limits<double>::quiet_NaN();
    double pre_pred_z = std::numeric_limits<double>::quiet_NaN();
    double pre_pred_yaw = std::numeric_limits<double>::quiet_NaN();
    double post_pred_x = std::numeric_limits<double>::quiet_NaN();
    double post_pred_y = std::numeric_limits<double>::quiet_NaN();
    double post_pred_z = std::numeric_limits<double>::quiet_NaN();
    double post_pred_yaw = std::numeric_limits<double>::quiet_NaN();

    double pre_dx = std::numeric_limits<double>::quiet_NaN();
    double pre_dy = std::numeric_limits<double>::quiet_NaN();
    double pre_dz = std::numeric_limits<double>::quiet_NaN();
    double pre_position_error = std::numeric_limits<double>::quiet_NaN();
    double post_dx = std::numeric_limits<double>::quiet_NaN();
    double post_dy = std::numeric_limits<double>::quiet_NaN();
    double post_dz = std::numeric_limits<double>::quiet_NaN();
    double post_position_error = std::numeric_limits<double>::quiet_NaN();
    double residual_radial = std::numeric_limits<double>::quiet_NaN();
    double residual_tangential = std::numeric_limits<double>::quiet_NaN();

    double center_x = std::numeric_limits<double>::quiet_NaN();
    double center_y = std::numeric_limits<double>::quiet_NaN();
    double center_z = std::numeric_limits<double>::quiet_NaN();
    double vx = std::numeric_limits<double>::quiet_NaN();
    double vy = std::numeric_limits<double>::quiet_NaN();
    double vz = std::numeric_limits<double>::quiet_NaN();
    double debug_ax = std::numeric_limits<double>::quiet_NaN();
    double debug_ay = std::numeric_limits<double>::quiet_NaN();

    double yaw = std::numeric_limits<double>::quiet_NaN();
    double w = std::numeric_limits<double>::quiet_NaN();
    double phase_w = std::numeric_limits<double>::quiet_NaN();
    double instant_phase_w = std::numeric_limits<double>::quiet_NaN();
    bool phase_valid = false;

    double r1 = std::numeric_limits<double>::quiet_NaN();
    double r2 = std::numeric_limits<double>::quiet_NaN();
    double h = std::numeric_limits<double>::quiet_NaN();
    double p_x = std::numeric_limits<double>::quiet_NaN();
    double p_vx = std::numeric_limits<double>::quiet_NaN();
    double p_y = std::numeric_limits<double>::quiet_NaN();
    double p_vy = std::numeric_limits<double>::quiet_NaN();
    double p_r1 = std::numeric_limits<double>::quiet_NaN();
    double p_r2 = std::numeric_limits<double>::quiet_NaN();
    double p_h = std::numeric_limits<double>::quiet_NaN();

    double nis = std::numeric_limits<double>::quiet_NaN();
    double nis_xyz = std::numeric_limits<double>::quiet_NaN();
    double nis_yaw = std::numeric_limits<double>::quiet_NaN();
    double yaw_variance_scale = 1.0;
    std::string tracker_state = "LOST";
    bool armor_switch = false;
    bool direction_reversal = false;
    bool pending_sign_conflict = false;
    bool association_success = false;
};

class EKFDebugPlotter {
public:
    explicit EKFDebugPlotter(const EKFDebugPlotConfig& config);

    bool active() const { return config_.enabled || config_.csv_enabled; }
    bool windowEnabled() const { return config_.enabled; }
    void update(EKFDebugPlotSample sample);
    cv::Mat render() const;
    const std::string& screenshotPath() const { return config_.screenshot_path; }
    bool saveLatestPng() const { return config_.save_latest_png; }

private:
    static double unwrapNear(double wrapped, double previous_unwrapped);
    void writeCsv(const EKFDebugPlotSample& sample);

    EKFDebugPlotConfig config_;
    std::deque<EKFDebugPlotSample> history_;
    std::ofstream csv_;
};
