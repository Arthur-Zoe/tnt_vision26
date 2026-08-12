// ArmorSolver.h
#ifndef ARMOR_SOLVER_H
#define ARMOR_SOLVER_H
#include <Eigen/Dense>


#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <opencv2/core/eigen.hpp> // 用于Eigen转换
#include "rclcpp/rclcpp.hpp"
#include <fstream> // <-- 添加文件流头文件
#include <memory>


#include "2d_armor_detector/LightBar.h"
#include "2d_armor_detector/Armor.h"
#include "ba_solver/ba_solver.hpp"
#include "ba_solver/utils.hpp"

#include <Eigen/Geometry> // For Quaternion and rotation matrix math


double getYawFromRvec(const cv::Mat& rvec);

std::vector<double> getNormalYawPitchRollFromRvec(const cv::Mat& rvec);


class ArmorSolver {
    
public:
    ArmorSolver(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node) {
        // 初始化相机参数
        initCameraMatrix(config_file_ptr, node);
        initArmorPoints();

        delta_x_ = (*config_file_ptr)["delta_x_"].as<float>();
        delta_y_ = (*config_file_ptr)["delta_y_"].as<float>();
        delta_z_ = (*config_file_ptr)["delta_z_"].as<float>();

        const YAML::Node yaw_config = (*config_file_ptr)["yaw_refinement"];
        yaw_refinement_.enabled = yaw_config["enabled"].as<bool>();
        constexpr double deg_to_rad = M_PI / 180.0;
        yaw_refinement_.search_half_range_rad =
            yaw_config["search_half_range_degree"].as<double>() * deg_to_rad;
        yaw_refinement_.coarse_step_rad =
            yaw_config["coarse_step_degree"].as<double>() * deg_to_rad;
        yaw_refinement_.fine_step_rad =
            yaw_config["fine_step_degree"].as<double>() * deg_to_rad;
        yaw_refinement_.max_accept_delta_rad =
            yaw_config["max_accept_delta_degree"].as<double>() * deg_to_rad;
        yaw_refinement_.min_rmse_improvement_px =
            yaw_config["min_rmse_improvement_px"].as<double>();
        yaw_refinement_.min_relative_improvement =
            yaw_config["min_relative_improvement"].as<double>();

    }
    // 新增3D到像素坐标投影函数
    cv::Point2f project3DToPixel(const cv::Point3f& world_point) const;

    AimResult solveArmor(const ArmorResult& armor_result,
                         double last_pitch_rad_,
                         double last_yaw_rad_,
                         double last_roll_rad_) const;
    
    
     /**
     * @brief 根据图像分辨率计算最大视场角（弧度）
     * @param width  图像宽度（像素）
     * @param height 图像高度（像素）
     * @return 最大夹角（弧度），若计算失败返回 -1.0
     */
    double getMaxFOVAngle(int width, int height) const;

private:
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;
    
    void initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    void initArmorPoints();
    rclcpp::Node* node;

    std::unique_ptr<fyt::auto_aim::BaSolver> ba_;

        // 设置logger
    rclcpp::Logger logger_p = rclcpp::get_logger("armor_solver");
    
    float delta_x_;
    float delta_y_;
    float delta_z_;

    struct YawRefinementConfig {
        bool enabled = false;
        double search_half_range_rad = 0.0;
        double coarse_step_rad = 0.0;
        double fine_step_rad = 0.0;
        double max_accept_delta_rad = 0.0;
        double min_rmse_improvement_px = 0.0;
        double min_relative_improvement = 0.0;
    } yaw_refinement_;
    // 用于缓存分辨率与对应最大夹角的映射（mutable 以便在 const 成员函数中修改）
    mutable std::unordered_map<std::string, double> fov_cache_;

    // 辅助函数：生成分辨率字符串键
    static std::string makeCacheKey(int width, int height) {
        return std::to_string(width) + "x" + std::to_string(height);
    }
};

#endif // ARMOR_SOLVER_H
