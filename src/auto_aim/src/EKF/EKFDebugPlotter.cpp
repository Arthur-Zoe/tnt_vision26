#include "EKF/EKFDebugPlotter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace {

template <typename T>
void readIfExists(const YAML::Node& node, const char* key, T& value) {
    if (node && node[key]) value = node[key].as<T>();
}

using Getter = double (*)(const EKFDebugPlotSample&);

struct Series {
    const char* label;
    cv::Scalar color;
    Getter get;
};

struct ReferenceLine {
    double value;
    cv::Scalar color;
    const char* label;
};

double finiteOrNaN(double value) {
    return std::isfinite(value)
        ? value : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

EKFDebugPlotConfig EKFDebugPlotConfig::fromYaml(const YAML::Node& root) {
    EKFDebugPlotConfig cfg;
    const YAML::Node plot = root["ekf_debug_plot"];
    readIfExists(plot, "enabled", cfg.enabled);
    readIfExists(plot, "csv_enabled", cfg.csv_enabled);
    readIfExists(plot, "save_latest_png", cfg.save_latest_png);
    readIfExists(plot, "history_seconds", cfg.history_seconds);
    readIfExists(plot, "width", cfg.width);
    readIfExists(plot, "panel_height", cfg.panel_height);
    readIfExists(plot, "csv_path", cfg.csv_path);
    readIfExists(plot, "screenshot_path", cfg.screenshot_path);

    const YAML::Node robust = root["robust_ekf"];
    readIfExists(robust, "association_nis_gate", cfg.nis_gate);
    const YAML::Node reference = robust["geometry_debug_reference"];
    readIfExists(reference, "r1", cfg.r1_ref);
    readIfExists(reference, "r2", cfg.r2_ref);
    readIfExists(reference, "h", cfg.h_ref);
    readIfExists(reference, "r1_tolerance", cfg.r1_tolerance);
    readIfExists(reference, "r2_tolerance", cfg.r2_tolerance);
    readIfExists(reference, "h_tolerance", cfg.h_tolerance);
    return cfg;
}

EKFDebugPlotter::EKFDebugPlotter(const EKFDebugPlotConfig& config)
    : config_(config) {
    config_.history_seconds = std::max(0.1, config_.history_seconds);
    config_.width = std::max(640, config_.width);
    config_.panel_height = std::max(80, config_.panel_height);
    if (config_.csv_enabled) {
        csv_.open(config_.csv_path, std::ios::out | std::ios::trunc);
        if (csv_) {
            csv_ << std::unitbuf;
            csv_ << "frame,timestamp,measurement_x,measurement_y,measurement_z,"
                    "measurement_yaw,matched_id,pre_pred_x,pre_pred_y,pre_pred_z,"
                    "pre_pred_yaw,post_pred_x,post_pred_y,post_pred_z,post_pred_yaw,"
                    "pre_dx,pre_dy,pre_dz,pre_position_error,post_dx,post_dy,post_dz,"
                    "post_position_error,center_x,center_y,center_z,vx,vy,vz,"
                    "debug_ax,debug_ay,yaw,w,phase_w,instant_phase_w,r1,r2,h,"
                    "P_x,P_vx,P_y,P_vy,P_r1,P_r2,P_h,NIS,NIS_xyz,NIS_yaw,"
                    "yaw_variance_scale,residual_radial,residual_tangential,"
                    "tracker_state,armor_switch,direction_reversal,"
                    "pending_sign_conflict,association_success\n";
        }
    }
}

double EKFDebugPlotter::unwrapNear(double wrapped, double previous_unwrapped) {
    if (!std::isfinite(wrapped) || !std::isfinite(previous_unwrapped)) {
        return wrapped;
    }
    constexpr double period = 2.0 * 3.14159265358979323846;
    return wrapped + std::round((previous_unwrapped - wrapped) / period) * period;
}

void EKFDebugPlotter::update(EKFDebugPlotSample sample) {
    if (!active() || !std::isfinite(sample.timestamp_s)) return;

    if (!history_.empty() && sample.timestamp_s <= history_.back().timestamp_s) {
        history_.clear();
    }
    if (!history_.empty()) {
        const EKFDebugPlotSample& previous = history_.back();
        const double dt = sample.timestamp_s - previous.timestamp_s;
        if (dt > 0.0 && std::isfinite(sample.vx) && std::isfinite(previous.vx)) {
            sample.debug_ax = (sample.vx - previous.vx) / dt;
        }
        if (dt > 0.0 && std::isfinite(sample.vy) && std::isfinite(previous.vy)) {
            sample.debug_ay = (sample.vy - previous.vy) / dt;
        }
        const EKFDebugPlotSample csv_sample = sample;
        auto lastFinite = [this](double EKFDebugPlotSample::*member) {
            for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
                const double value = (*it).*member;
                if (std::isfinite(value)) return value;
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
        sample.yaw = unwrapNear(
            sample.yaw, lastFinite(&EKFDebugPlotSample::yaw));
        constexpr double half_pi = 3.14159265358979323846 / 2.0;
        const double matched_branch =
            std::isfinite(sample.yaw) && sample.matched_id >= 0
                ? sample.yaw + sample.matched_id * half_pi
                : lastFinite(&EKFDebugPlotSample::pre_pred_yaw);
        sample.pre_pred_yaw = unwrapNear(
            sample.pre_pred_yaw, matched_branch);
        sample.measurement_yaw = unwrapNear(
            sample.measurement_yaw, sample.pre_pred_yaw);
        sample.post_pred_yaw = unwrapNear(
            sample.post_pred_yaw, sample.pre_pred_yaw);
        writeCsv(csv_sample);
    } else {
        writeCsv(sample);
    }
    history_.push_back(std::move(sample));
    const double cutoff = history_.back().timestamp_s - config_.history_seconds;
    while (!history_.empty() && history_.front().timestamp_s < cutoff) {
        history_.pop_front();
    }
}

void EKFDebugPlotter::writeCsv(const EKFDebugPlotSample& s) {
    if (!csv_) return;
    csv_ << s.frame_id << ',' << std::setprecision(12) << s.timestamp_s << ','
         << s.measurement_x << ',' << s.measurement_y << ',' << s.measurement_z << ','
         << s.measurement_yaw << ',' << s.matched_id << ','
         << s.pre_pred_x << ',' << s.pre_pred_y << ',' << s.pre_pred_z << ','
         << s.pre_pred_yaw << ',' << s.post_pred_x << ',' << s.post_pred_y << ','
         << s.post_pred_z << ',' << s.post_pred_yaw << ','
         << s.pre_dx << ',' << s.pre_dy << ',' << s.pre_dz << ','
         << s.pre_position_error << ',' << s.post_dx << ',' << s.post_dy << ','
         << s.post_dz << ',' << s.post_position_error << ','
         << s.center_x << ',' << s.center_y << ',' << s.center_z << ','
         << s.vx << ',' << s.vy << ',' << s.vz << ','
         << s.debug_ax << ',' << s.debug_ay << ',' << s.yaw << ',' << s.w << ','
         << s.phase_w << ',' << s.instant_phase_w << ','
         << s.r1 << ',' << s.r2 << ',' << s.h << ','
         << s.p_x << ',' << s.p_vx << ',' << s.p_y << ',' << s.p_vy << ','
         << s.p_r1 << ',' << s.p_r2 << ',' << s.p_h << ','
         << s.nis << ',' << s.nis_xyz << ',' << s.nis_yaw << ','
         << s.yaw_variance_scale << ',' << s.residual_radial << ','
         << s.residual_tangential << ',' << s.tracker_state << ','
         << (s.armor_switch ? 1 : 0) << ','
         << (s.direction_reversal ? 1 : 0) << ','
         << (s.pending_sign_conflict ? 1 : 0) << ','
         << (s.association_success ? 1 : 0) << '\n';
}

cv::Mat EKFDebugPlotter::render() const {
    constexpr int panel_count = 9;
    cv::Mat canvas(config_.panel_height * panel_count, config_.width,
                   CV_8UC3, cv::Scalar(20, 20, 20));
    if (history_.empty()) return canvas;

    const int left = 92;
    const int right = 16;
    const int top_margin = 20;
    const int bottom_margin = 18;
    const double t_end = history_.back().timestamp_s;
    const double t_start = t_end - config_.history_seconds;

    auto drawPanel = [&](int index, const char* title,
                         const std::vector<Series>& series,
                         const std::vector<ReferenceLine>& references = {}) {
        const int y0 = index * config_.panel_height;
        const cv::Rect plot(left, y0 + top_margin,
                            config_.width - left - right,
                            config_.panel_height - top_margin - bottom_margin);
        cv::rectangle(canvas, plot, cv::Scalar(65, 65, 65), 1);

        double ymin = std::numeric_limits<double>::infinity();
        double ymax = -std::numeric_limits<double>::infinity();
        for (const auto& sample : history_) {
            for (const auto& s : series) {
                const double value = finiteOrNaN(s.get(sample));
                if (std::isfinite(value)) {
                    ymin = std::min(ymin, value);
                    ymax = std::max(ymax, value);
                }
            }
        }
        for (const auto& ref : references) {
            if (std::isfinite(ref.value)) {
                ymin = std::min(ymin, ref.value);
                ymax = std::max(ymax, ref.value);
            }
        }
        if (!std::isfinite(ymin) || !std::isfinite(ymax)) {
            ymin = -1.0;
            ymax = 1.0;
        }
        double span = ymax - ymin;
        if (span < 1e-9) span = std::max(1.0, std::abs(ymax) * 0.2);
        ymin -= span * 0.08;
        ymax += span * 0.08;

        auto px = [&](double t) {
            return plot.x + static_cast<int>(
                std::clamp((t - t_start) / config_.history_seconds, 0.0, 1.0) *
                plot.width);
        };
        auto py = [&](double value) {
            return plot.y + plot.height - static_cast<int>(
                std::clamp((value - ymin) / (ymax - ymin), 0.0, 1.0) *
                plot.height);
        };

        for (const auto& ref : references) {
            if (std::isfinite(ref.value)) {
                cv::line(canvas, cv::Point(plot.x, py(ref.value)),
                         cv::Point(plot.x + plot.width, py(ref.value)),
                         ref.color, 1, cv::LINE_AA);
            }
        }

        for (const auto& s : series) {
            std::vector<cv::Point> segment;
            for (const auto& sample : history_) {
                const double value = finiteOrNaN(s.get(sample));
                if (!std::isfinite(value)) {
                    if (segment.size() >= 2) cv::polylines(canvas, segment, false, s.color, 1, cv::LINE_AA);
                    segment.clear();
                    continue;
                }
                segment.emplace_back(px(sample.timestamp_s), py(value));
            }
            if (segment.size() >= 2) cv::polylines(canvas, segment, false, s.color, 1, cv::LINE_AA);
        }

        if (std::string(title) == "angular velocity") {
            for (const auto& sample : history_) {
                cv::Scalar color;
                bool event = true;
                if (sample.direction_reversal) color = cv::Scalar(0, 0, 255);
                else if (sample.pending_sign_conflict) color = cv::Scalar(0, 165, 255);
                else if (sample.armor_switch) color = cv::Scalar(255, 0, 255);
                else event = false;
                if (event) cv::line(canvas, cv::Point(px(sample.timestamp_s), plot.y),
                                    cv::Point(px(sample.timestamp_s), plot.y + plot.height), color, 1);
            }
        }

        cv::putText(canvas, title, cv::Point(5, y0 + 17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(230, 230, 230), 1);
        cv::putText(canvas, cv::format("%.3g", ymax), cv::Point(4, plot.y + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(170, 170, 170), 1);
        cv::putText(canvas, cv::format("%.3g", ymin),
                    cv::Point(4, plot.y + plot.height), cv::FONT_HERSHEY_SIMPLEX,
                    0.35, cv::Scalar(170, 170, 170), 1);
        int legend_x = left + 8;
        for (const auto& s : series) {
            cv::putText(canvas, s.label, cv::Point(legend_x, y0 + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, s.color, 1);
            legend_x += static_cast<int>(std::string(s.label).size()) * 8 + 18;
        }
    };

#define FIELD_GETTER(name, field) \
    static const auto name = [](const EKFDebugPlotSample& s) -> double { return s.field; }
    FIELD_GETTER(meas_x, measurement_x); FIELD_GETTER(pre_x, pre_pred_x); FIELD_GETTER(post_x, post_pred_x);
    FIELD_GETTER(meas_y, measurement_y); FIELD_GETTER(pre_y, pre_pred_y); FIELD_GETTER(post_y, post_pred_y);
    FIELD_GETTER(center_x, center_x); FIELD_GETTER(center_y, center_y);
    FIELD_GETTER(vx, vx); FIELD_GETTER(vy, vy);
    FIELD_GETTER(meas_yaw, measurement_yaw); FIELD_GETTER(pre_yaw, pre_pred_yaw); FIELD_GETTER(base_yaw, yaw);
    FIELD_GETTER(w, w); FIELD_GETTER(phase_w, phase_w); FIELD_GETTER(inst_w, instant_phase_w);
    FIELD_GETTER(r1, r1); FIELD_GETTER(r2, r2); FIELD_GETTER(h, h);
    FIELD_GETTER(pre_err, pre_position_error); FIELD_GETTER(post_err, post_position_error);
    FIELD_GETTER(radial, residual_radial); FIELD_GETTER(tangential, residual_tangential);
    FIELD_GETTER(nis, nis); FIELD_GETTER(nis_xyz, nis_xyz); FIELD_GETTER(nis_yaw, nis_yaw);
#undef FIELD_GETTER

    drawPanel(0, "armor x [m]", {{"meas", {0,255,255}, meas_x}, {"pre", {0,128,255}, pre_x}, {"post", {0,255,0}, post_x}});
    drawPanel(1, "armor y [m]", {{"meas", {0,255,255}, meas_y}, {"pre", {0,128,255}, pre_y}, {"post", {0,255,0}, post_y}});
    drawPanel(2, "center [m]", {{"x", {255,180,0}, center_x}, {"y", {180,0,255}, center_y}});
    drawPanel(3, "velocity [m/s]", {{"vx", {255,180,0}, vx}, {"vy", {180,0,255}, vy}});
    drawPanel(4, "yaw unwrapped [rad]", {{"meas armor", {0,255,255}, meas_yaw}, {"matched pre", {0,128,255}, pre_yaw}, {"base", {0,255,0}, base_yaw}});
    drawPanel(5, "angular velocity", {{"EKF w", {0,255,0}, w}, {"phase w", {255,180,0}, phase_w}, {"instant", {0,128,255}, inst_w}});
    drawPanel(6, "geometry [m]", {{"r1", {0,255,255}, r1}, {"r2", {255,180,0}, r2}, {"h", {180,0,255}, h}},
              {{config_.r1_ref, {0,130,130}, "r1 ref"}, {config_.r1_ref-config_.r1_tolerance, {0,70,70}, ""}, {config_.r1_ref+config_.r1_tolerance, {0,70,70}, ""},
               {config_.r2_ref, {130,80,0}, "r2 ref"}, {config_.r2_ref-config_.r2_tolerance, {70,40,0}, ""}, {config_.r2_ref+config_.r2_tolerance, {70,40,0}, ""},
               {config_.h_ref, {80,0,130}, "h ref"}, {config_.h_ref-config_.h_tolerance, {45,0,70}, ""}, {config_.h_ref+config_.h_tolerance, {45,0,70}, ""}});
    drawPanel(7, "position residual [m]", {{"pre", {0,0,255}, pre_err}, {"post", {0,255,0}, post_err}, {"radial", {0,255,255}, radial}, {"tangent", {255,180,0}, tangential}});
    drawPanel(8, "NIS", {{"total", {0,0,255}, nis}, {"xyz", {0,255,255}, nis_xyz}, {"yaw", {255,180,0}, nis_yaw}}, {{config_.nis_gate, {80,80,255}, "gate"}});
    return canvas;
}
