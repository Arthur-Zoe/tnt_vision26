// ArmorDetect_Node.cpp
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "camera/Camera.h"
#include "2d_armor_detector/LightBarDetector.h"
#include "2d_armor_detector/ArmorDetector.h"
#include "2d_armor_detector/ArmorClassifier.h"
#include "3d_processing/ArmorSolver.h"
//#include "armor_detector/ArmorAngleKalman.h"

//#include "auto_aim/msg/serial_data.hpp"
//#include "auto_aim/msg/gimbal_command.hpp"
#include <chrono>
#include <string>
#include <thread>
#include <3d_processing/BallisticSolver.h>
#include <yaml-cpp/yaml.h>
#include "utils/FrameRateCounter.h"
#include "2d_armor_detector/UnwarpUtils.h"
#include "other_input/VideoInput.h"
#include "other_input/ImagesInput.h"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include <queue>
#include <fstream>
#include <iomanip>
#include "communication/Com.h"
#include <csignal>
#include "3d_processing/RestFrame.h"
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include "predictor/PredictorMain.h"
// #include "2d_armor_detector/YOLOPoseArmorDetector.h"
#include "2d_armor_detector/Armor.h"
#include "communication/WatchdogClient.h"
#include "visualizer/RestFrameDraw.h"
#include "communication/HeadIMU.h"
#include "visualizer/YawVisualizer.h"
#include "logger/TwoVideoLogger.h"
#include "RP24_YOLO/RP24_YOLO_Wrapper.h"
#include "EKF/EKFDebugPlotter.h"

namespace fs = std::filesystem;

#include "macro/AutoAimMacro.h"

// 全局变量定义
FramePacket g_frame_packet;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
bool g_bExit = false;
bool image_used = true;

class ArmorDetectNode : public rclcpp::Node {
public:
    ArmorDetectNode() : Node("armor_detect_node") {
        node_start_time = std::chrono::steady_clock::now();

        // 1. 获取可执行文件路径    
        char exec_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
        if (len == -1) {
            perror("readlink");
            return;
        }
        exec_path[len] = '\0';
        RCLCPP_INFO(this->get_logger(), "info from C++ | Path: %s\n", exec_path);
        // 2. 转换为文件系统路径对象
        fs::path full_path(exec_path);
        std::string full_path_str = full_path.string();  // 转换为字符串便于查找
        // 3. 查找工作空间目录名
        const std::string ws_dir_name = "transistor_rm2026_algorithm_visual_ws";
        size_t pos = full_path_str.find(ws_dir_name);
        if (pos == std::string::npos) {
            std::cerr << "Error: Workspace directory not found in path" << std::endl;
            return;
        }
        // 4. 截取到工作空间目录结尾
        fs::path ws_dir_path = full_path_str.substr(0, pos + ws_dir_name.length());
        // 5. 拼接模型路径
        const std::string config_file_relatvie_path = "src/shared_files/config.yaml";
        fs::path config_file_path = ws_dir_path / config_file_relatvie_path;  // 使用文件系统的路径拼接

        // 加载配置文件
        config_file_ptr = std::make_shared<YAML::Node>(YAML::LoadFile(config_file_path));
        ekf_debug_plotter_ = std::make_unique<EKFDebugPlotter>(
            EKFDebugPlotConfig::fromYaml(*config_file_ptr));

        const YAML::Node yaw_debug_config =
            (*config_file_ptr)["yaw_refinement"];
        yaw_debug_csv_enabled_ = yaw_debug_config["debug_csv"].as<bool>();
        if (yaw_debug_csv_enabled_) {
            yaw_debug_csv_.open(
                yaw_debug_config["debug_csv_path"].as<std::string>(),
                std::ios::out | std::ios::trunc);
            if (yaw_debug_csv_) {
                yaw_debug_csv_
                    << "frame_id,timestamp_s,target_type,measurement_number,"
                       "x_mm,y_mm,z_mm,yaw_raw_rad,yaw_refined_rad,"
                       "yaw_used_rad,yaw_delta_rad,repr_raw_px,repr_refined_px,"
                       "facing_angle_rad,refined_valid,refinement_status,"
                       "ekf_yaw_rad,ekf_w_rad_s,nis,ekf_state,target_state,"
                       "matched_armor_id,armor_switched\n";
            } else {
                yaw_debug_csv_enabled_ = false;
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to open yaw debug CSV");
            }
        }

        const YAML::Node pnp_debug_config =
            (*config_file_ptr)["pnp_diagnostic"];
        pnp_debug_csv_enabled_ =
            pnp_debug_config["csv_enabled"].as<bool>();
        calibration_resolution_known_ =
            pnp_debug_config["calibration_resolution_known"].as<bool>();
        calibration_width_ = pnp_debug_config["calibration_width"].as<int>();
        calibration_height_ = pnp_debug_config["calibration_height"].as<int>();
        detector_input_width_ =
            pnp_debug_config["detector_input_width"].as<int>();
        detector_input_height_ =
            pnp_debug_config["detector_input_height"].as<int>();
        if (pnp_debug_csv_enabled_) {
            pnp_debug_csv_.open(
                pnp_debug_config["csv_path"].as<std::string>(),
                std::ios::out | std::ios::trunc);
            if (pnp_debug_csv_) {
                pnp_debug_csv_ << std::unitbuf;
                pnp_debug_csv_
                    << "frame_id,timestamp_s,detection_index,armor_number,is_large,"
                       "is_tracked_now,frame_width,frame_height,"
                       "calibration_resolution_known,calibration_width,"
                       "calibration_height,detector_input_width,"
                       "detector_input_height,detector_scale_x,detector_scale_y,"
                       "current_corner_length_scale,object_width_mm,object_height_mm";
                for (const char* corner_space : {"raw", "current"}) {
                    for (int corner = 0; corner < 4; ++corner) {
                        pnp_debug_csv_ << ',' << corner_space << "_c" << corner << "_x_px"
                                       << ',' << corner_space << "_c" << corner << "_y_px";
                    }
                }
                const auto write_pose_header = [this](const std::string& prefix) {
                    pnp_debug_csv_
                        << ',' << prefix << "_valid"
                        << ',' << prefix << "_camera_x_m"
                        << ',' << prefix << "_camera_y_m"
                        << ',' << prefix << "_camera_z_m"
                        << ',' << prefix << "_world_x_m"
                        << ',' << prefix << "_world_y_m"
                        << ',' << prefix << "_world_z_m"
                        << ',' << prefix << "_range_m"
                        << ',' << prefix << "_reprojection_rmse_px"
                        << ',' << prefix << "_world_yaw_rad"
                        << ',' << prefix << "_facing_angle_rad";
                };
                write_pose_header("raw");
                write_pose_header("current");
                for (const char* scale : {"s090", "s095", "s100", "s105", "s110"}) {
                    write_pose_header(scale);
                }
                pnp_debug_csv_ << '\n';
            } else {
                pnp_debug_csv_enabled_ = false;
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to open PnP diagnostic CSV");
            }
        }

        const YAML::Node geometry_debug_config =
            (*config_file_ptr)["robust_ekf"]["geometry"];
        geometry_debug_csv_enabled_ =
            geometry_debug_config["debug_csv"].as<bool>();
        if (geometry_debug_csv_enabled_) {
            geometry_debug_csv_.open(
                geometry_debug_config["debug_csv_path"].as<std::string>(),
                std::ios::out | std::ios::trunc);
            if (geometry_debug_csv_) {
                geometry_debug_csv_ << std::unitbuf;
                geometry_debug_csv_
                    << "frame_id,timestamp_s,target_type,measurement_number,"
                       "has_measurement,target_state,ekf_state,updated,"
                       "measurement_valid,current_armor_id,r1_m,r2_m,h_m,"
                       "p_r1_m2,p_r2_m2,p_h_m2,center_x_m,center_y_m,"
                       "center_z_m,state_yaw_rad,w_rad_s,nis,matched_armor_id,"
                       "armor_parity,armor_switched,direction_reversal,"
                       "pending_sign_conflict,recovered,geometry_valid,"
                       "geometry_update_allowed,geometry_preserved\n";
            } else {
                geometry_debug_csv_enabled_ = false;
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to open geometry debug CSV");
            }
        }

        association_debug_csv_enabled_ =
            geometry_debug_config["association_debug_csv"].as<bool>();
        if (association_debug_csv_enabled_) {
            association_debug_csv_.open(
                geometry_debug_config["association_debug_csv_path"]
                    .as<std::string>(),
                std::ios::out | std::ios::trunc);
            if (association_debug_csv_) {
                association_debug_csv_ << std::unitbuf;
                association_debug_csv_
                    << "frame_id,timestamp_s,target_type,target_state,ekf_state,"
                       "tracker_state_before,measurement_number,current_armor_id,"
                       "best_id,candidate_is_switch,armor_switched,"
                       "temp_lost_recovery,recovered,phase_valid,phase_delta,"
                       "phase_w,pending_sign_conflict,direction_reversal,"
                       "topology_event,measurement_yaw,best_predicted_yaw,"
                       "best_yaw_innovation,hypothetical_best_nis,"
                       "hypothetical_best_nis_x,hypothetical_best_nis_y,"
                       "hypothetical_best_nis_z,hypothetical_best_nis_yaw,"
                       "hypothesis_id,"
                       "is_current,selected,center_x_m,center_y_m,center_z_m,"
                       "state_yaw_rad,r1_m,r2_m,h_m,p_r1_m2,p_r2_m2,p_h_m2,"
                       "predicted_x_m,predicted_y_m,predicted_z_m,"
                       "predicted_yaw_rad,facing_angle_rad,range_pass,"
                       "visibility_pass,innovation_x_m,innovation_y_m,"
                       "innovation_z_m,innovation_yaw_rad,position_error_m,"
                       "yaw_error_rad,association_yaw_variance_scale,"
                       "nis,nis_x,nis_y,nis_z,nis_yaw,"
                       "hypothetical_scaled_nis,hypothetical_scaled_nis_x,"
                       "hypothetical_scaled_nis_y,hypothetical_scaled_nis_z,"
                       "hypothetical_scaled_nis_yaw,"
                       "fixed_radius_cov_nis,fixed_radius_cov_nis_x,"
                       "fixed_radius_cov_nis_y,fixed_radius_cov_nis_z,"
                       "fixed_radius_cov_nis_yaw,"
                       "nis_gate_pass,position_gate_pass,yaw_gate_pass,"
                       "passes_all_measurement_gates,radial_residual_m,"
                       "tangential_residual_m,updated\n";
            } else {
                association_debug_csv_enabled_ = false;
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to open association debug CSV");
            }
        }

        lifecycle_debug_csv_enabled_ =
            geometry_debug_config["lifecycle_debug_csv"].as<bool>();
        if (lifecycle_debug_csv_enabled_) {
            lifecycle_debug_csv_.open(
                geometry_debug_config["lifecycle_debug_csv_path"]
                    .as<std::string>(),
                std::ios::out | std::ios::trunc);
            if (lifecycle_debug_csv_) {
                lifecycle_debug_csv_ << std::unitbuf;
                lifecycle_debug_csv_
                    << "frame_id,timestamp_s,target_state,target_type,"
                       "has_measurement,measurement_number,ekf_state,"
                       "ekf_updated,armor_switched,nis\n";
            } else {
                lifecycle_debug_csv_enabled_ = false;
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to open lifecycle debug CSV");
            }
        }



        // 初始化参数
        
#ifdef FIX_ENEMY_COLOR
        enemy_color_ = (FIX_ENEMY_COLOR == 0) ? "RED" : "BLUE";
#else
        enemy_color_ = (*config_file_ptr)["init_enemy_color"].as<std::string>();
#endif
#ifdef FIX_BULLET_VELOCITY
        bullet_velocity_ = FIX_BULLET_VELOCITY;
#else
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();
#endif

        use_RP24_YOLO = (*config_file_ptr)["use_RP24_YOLO"].as<bool>();

        // 根据相机内参自动提取，不再需要手动输入
        // yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        // pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 
        const YAML::Node& camera_matrix_Node = (*config_file_ptr)["camera_matrix"];
        yaw_rad_to_x_pixel_ratio = camera_matrix_Node[0][0].as<float>(); 
        pitch_rad_to_y_pixel_ratio = camera_matrix_Node[1][1].as<float>(); 


        max_armor_position_height = (*config_file_ptr)["max_armor_position_height"].as<float>(); 
        
        params_.min_light_height = (*config_file_ptr)["min_light_height"].as<int>();
        params_.light_min_area = (*config_file_ptr)["light_min_area"].as<int>();
        params_.light_max_area = (*config_file_ptr)["light_max_area"].as<int>();
        params_.max_light_wh_ratio = (*config_file_ptr)["max_light_wh_ratio"].as<float>();
        params_.min_light_wh_ratio = (*config_file_ptr)["min_light_wh_ratio"].as<float>();
        params_.light_max_tilt_angle = (*config_file_ptr)["light_max_tilt_angle"].as<float>();
        
        frame_rate_ = (*config_file_ptr)["frame_rate"].as<float>();


#ifdef USE_VIDEO
        video_input_ = std::make_shared<VideoInput>(
            ws_dir_path / (*config_file_ptr)["video_relative_path"].as<std::string>(),
            frame_rate_);
#else
#ifdef USE_IMAGES
        images_input_ = std::make_shared<ImagesInput>(
            ws_dir_path / (*config_file_ptr)["images_relative_path"].as<std::string>(),
            frame_rate_);
#else
        // 初始化相机和检测器
        camera_ = std::make_shared<Camera>((*config_file_ptr)["cam_ip"].as<std::string>(), (*config_file_ptr)["pc_ip"].as<std::string>());
        //camera_ = std::make_shared<Camera>(0);
        camera_->setExposureTime((*config_file_ptr)["camera_ExposureTime"].as<float>());
        camera_->setGain((*config_file_ptr)["camera_Gain"].as<float>());
        camera_ -> start();
#endif
#endif
        serial_delay_time = (*config_file_ptr)["serial_delay_time"].as<float>();

        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        } else if (enemy_color_ == "GREEN") {
            params_.enemy_color = Params::GREEN;
        } else if (enemy_color_ == "BOTH") {
            params_.enemy_color = Params::BOTH;
        } else {
            // 处理错误情况，设置默认值
            enemy_color_ = "GREEN";
            params_.enemy_color = Params::GREEN;
        }

        light_detector_ = std::make_shared<LightBarDetector>(params_, config_file_ptr, this);
        armor_detector_ = std::make_shared<ArmorDetector>(config_file_ptr, this);
        classifier_ = std::make_shared<ArmorClassifier>(config_file_ptr, this, ws_dir_path);
        armor_solver_ = std::make_shared<ArmorSolver>(config_file_ptr, this);
        ballistic_solver_ = std::make_shared<BallisticSolver>(config_file_ptr, this);

        rest_frame_ = std::make_shared<RestFrame>();
        rest_frame_ -> updateCamOrientation(0, 0, 0);
        rest_frame_ -> updateCamPosition(0, 0, 0);

        fps_counter = std::make_shared<FrameRateCounter>(30); // 30帧滑动窗口统计帧率

        predictor_main_ = std::make_shared<PredictorMain>(
            config_file_ptr, this, node_start_time, armor_solver_,
            ballistic_solver_, rest_frame_, fps_counter);

        rp24_yolo_wrapper = std::make_shared<RP24YOLOWrapper>(config_file_ptr, this, 
            ws_dir_path / (*config_file_ptr)["RP24_YOLO_model_relative_path"].as<std::string>(), 
            (*config_file_ptr)["RP24_YOLO_device"].as<std::string>());

        yaw_visualizer_ = std::make_shared<YawVisualizer>();

        com_data_visualize_frame = cv::Mat::zeros(480, 640, CV_8UC3);
#if (defined LOG_RESULT_VIDEO) or (defined LOG_ORIGIN_VIDEO)
        two_video_logger = std::make_shared<TwoVideoLogger>(ws_dir_path / "VideoLog");
#endif

        DelayInfos init_serial_infos;
        init_serial_infos.last_pitch_rad_ = 0.0;
        init_serial_infos.last_yaw_rad_ = 0.0;
        init_serial_infos.total_yaw_rad_ = 0.0;
        init_serial_infos.last_roll_rad_ = 0.0;
        init_serial_infos.push_time = node_start_time;
        serial_infos_delay_.push(init_serial_infos);

        // 初始化串口通信器
        serial_communication_ = std::make_shared<SerialCommunicationClass>(this, std::bind(&ArmorDetectNode::serialDataCallback, this, std::placeholders::_1));

        com_timer_thread_ = std::thread(std::bind(&SerialCommunicationClass::timerThread, serial_communication_));
        // com_timer_thread_.detach();

        headIMUInfos.headIMU_communication_ = std::make_shared<HeadIMUSerialCommunicationClass>(std::bind(&ArmorDetectNode::headIMUSerialDataCallback, this, std::placeholders::_1));
        headIMUInfos.headIMU_timer_thread_ = std::thread(std::bind(&HeadIMUSerialCommunicationClass::timerThread, headIMUInfos.headIMU_communication_));

        // 串口通信下位机初始化
        serial_communication_->sendData(0, 0, false);

        watchdog_client = std::make_shared<WatchdogClient>();
        watchdog_client -> init();
        watchdog_client -> feed();
        last_feed_dog_time = std::chrono::steady_clock::now();

#ifdef DEBUG_CODE
        debug_code();
#endif

        // // 创建定时器
        // timer_ = this->create_wall_timer(
        //     std::chrono::milliseconds((int)(1000/frame_rate_)), // 33
        //     std::bind(&ArmorDetectNode::processImage, this));
        main_loop_thread_ = std::thread(std::bind(&ArmorDetectNode::main_loop_func, this));


        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode initialized");
    }

    ~ArmorDetectNode() {
        serial_communication_->~SerialCommunicationClass();
        cv::destroyAllWindows();
        pthread_mutex_destroy(&g_mutex);
        RCLCPP_INFO(this->get_logger(), "ArmorDetectNode destroyed");
    }

private:
    void main_loop_func() {
        while (true) {
            std::chrono::steady_clock::time_point loop_start_time = std::chrono::steady_clock::now();
            processImage();
            std::this_thread::sleep_until(loop_start_time + std::chrono::microseconds(static_cast<int>(1e6 / frame_rate_)));
        }
    }

    void debug_code() {
        // while (true) {
        //     static double debug_time_count = 0.0;
        //     double debug_freq = 0.3;
        //     double debug_yaw = std::cos(debug_time_count*M_PI*debug_freq) * M_PI / 6;
        //     double debug_pitch = std::sin(debug_time_count*M_PI*debug_freq) * M_PI / 6;
        //     serial_communication_->sendData(debug_pitch, debug_yaw);
        //     RCLCPP_INFO(this->get_logger(), "send debug data: yaw[%.2f] pitch[%.2f]", debug_yaw, debug_pitch);
        //     RCLCPP_INFO(this->get_logger(), "received data: yaw[%.2f] pitch[%.2f]", last_yaw_rad_delayed_, last_pitch_rad_delayed_);
        //     cv::Mat frame;
        //     pthread_mutex_lock(&g_mutex);
        //     if (!g_frame_packet.image.empty()) {
        //         frame = g_frame_packet.image.clone();
        //         image_used = true;
        //     }
        //     pthread_mutex_unlock(&g_mutex);
        //     if (!frame.empty()) {
        //         cv::imshow("debug_code", frame);
        //         cv::waitKey(1);
        //     }
        //     auto start = std::chrono::steady_clock::now();
        //     std::this_thread::sleep_until(start + std::chrono::microseconds(33000));
        //     debug_time_count += 0.033;
        // }
        std::thread([&]() {
            double debug_time_count = 0.0;
            while (true) {
                auto start = std::chrono::steady_clock::now();

                SerialData fakeSerialData;
                fakeSerialData.bullet_velocity = 25.0;  // 子弹速度
                fakeSerialData.bullet_angle = std::sin(debug_time_count * 0.5 * (2*M_PI)) * 1.8 / 30 * 15;    // 子弹角度
                fakeSerialData.gimbal_yaw =  
                    // static_cast<int16_t>(60.0 * 4095.0 / 180.0);
                    // static_cast<int16_t>(std::atan2(std::sin(debug_time_count * 2 * M_PI), std::cos(debug_time_count * 2 * M_PI)) * 4095.0 / M_PI / 12); 
                    // static_cast<int16_t>(static_cast<float>((std::atan2(std::sin(debug_time_count * 1.0), std::cos(debug_time_count * 1.0)) > 0) - 0.5) * 4095); 
                    static_cast<int16_t>(std::atan2(std::sin(debug_time_count * 0.3), std::cos(debug_time_count * 0.3)) * 4095.0 / M_PI);
                    // static_cast<int16_t>(std::cos(debug_time_count * 0.5 * (2*M_PI)) * 4095 / 180 * 15);       // 云台当前偏航角
                fakeSerialData.color = 1;            // 敌方颜色(0:红色, 1:蓝色)

                serialDataCallback(fakeSerialData);

                std::this_thread::sleep_until(start + std::chrono::microseconds(10000));  // 大约10ms周期
                debug_time_count += 0.01;
            }
        }).detach();
    }

    void recalibrateHeadIMU() {
        float start_yaw = last_yaw_rad_imu_ + headIMUInfos.to_mcu_delta_yaw;
        float start_pitch = last_pitch_rad_delayed_;

        if (predictor_main_) {
            predictor_main_ -> reset_yaw_integration();
        }

        for (int i = 0; i < 20; i++) {
            serial_communication_ -> sendData(0.0, start_yaw, false);
            usleep(30*1000);
        }

        float new_yaw = last_yaw_rad_imu_;

        float delta_yaw = new_yaw - start_yaw;

        headIMUInfos.to_mcu_delta_yaw = -delta_yaw;

        serial_communication_ -> sendData(start_pitch, start_yaw + headIMUInfos.to_mcu_delta_yaw, false);
    }

    void headIMUSerialDataCallback(const HeadIMUSerialData& msg) {


        float current_pitch_;
        float current_yaw_;
        float current_roll_;
        float last_pitch_rad_;
        float last_yaw_rad_;
        float total_yaw_rad_;


        current_pitch_ = msg.euler_pitch;
        current_yaw_ = msg.euler_yaw;
        current_roll_ = msg.euler_roll;

        headIMUInfos.head_imu_yaw = msg.euler_yaw;
        headIMUInfos.head_imu_pitch = msg.euler_pitch;
        headIMUInfos.head_imu_roll = msg.euler_roll;
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;

        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        
        if (current_yaw_ < -M_PI/2 && last_yaw_rad_imu_ > M_PI/2) {
            current_yaw_circle_imu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_imu_ < -M_PI/2) {
            current_yaw_circle_imu_ -= 1;
        }

        total_yaw_rad_imu_ = current_yaw_circle_imu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_imu_ = current_pitch_;
        last_yaw_rad_imu_ = current_yaw_;
        last_roll_rad_imu_ = current_roll_;

        if (headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_imu_; // last_pitch_rad_mcu_
            now_serial_infos.last_roll_rad_ = last_roll_rad_imu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_imu_;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_imu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    void serialDataCallback(const SerialData& msg) {
        if (com_data_visualize_frame_used) {
            const MCUDataFrame& odf = msg.origin_data_frame;
            com_data_visualize_frame.setTo(cv::Scalar(0, 0, 0));
            cv::putText(com_data_visualize_frame, 
                cv::format("bullet_velocity: %.6f", odf.bullet_velocity), 
                cv::Point(20, 20),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("bullet_angle: %.6f", odf.bullet_angle), 
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("gimbal_yaw: %d", odf.gimbal_yaw), 
                cv::Point(20, 80),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("mark: %u", odf.mark), 
                cv::Point(20, 110),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("color: %u", odf.color), 
                cv::Point(20, 140),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(com_data_visualize_frame, 
                cv::format("z_rotation_velocity: %.6f", odf.z_rotation_velocity), 
                cv::Point(20, 170),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            com_data_visualize_frame_used = false;
        }


        SerialData processed_msg = msg;
#ifdef FIX_ENEMY_COLOR
        processed_msg.color = FIX_ENEMY_COLOR;
#endif
#ifdef FIX_BULLET_VELOCITY
        processed_msg.bullet_velocity = FIX_BULLET_VELOCITY;
#endif


        float current_pitch_;
        float current_yaw_;


        bullet_velocity_ = processed_msg.bullet_velocity;
        current_pitch_ = ((float)(processed_msg.bullet_angle)) * 30 / 1.8 * M_PI / 180; // 测定pitch轴传入数据1.8大约对应30°
        current_yaw_ = ((float)(processed_msg.gimbal_yaw)) * M_PI / 4096.0;  // 一圈对应[-4096, 4095]


        headIMUInfos.mcu_yaw = current_yaw_;
        headIMUInfos.mcu_pitch = current_pitch_;
        if (headIMUInfos.last_mcu_yaw != headIMUInfos.mcu_yaw) {
            headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update = headIMUInfos.head_imu_yaw;
            headIMUInfos.last_mcu_yaw = current_yaw_;
            headIMUInfos.last_mcu_yaw_update_time = std::chrono::steady_clock::now();
            headIMUInfos.mcu_yaw_online = true;
            headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update = headIMUInfos.last_mcu_command_yaw;
            headIMUInfos.to_mcu_delta_yaw = headIMUInfos.mcu_yaw - headIMUInfos.latest_head_imu_yaw_when_mcu_yaw_update;
        }
        headIMUInfos.to_mcu_delta_pitch = headIMUInfos.mcu_pitch - headIMUInfos.head_imu_pitch;


        while (current_yaw_ < -M_PI) {
            current_yaw_ += 2 * M_PI;
        }
        while (current_yaw_ > M_PI) {
            current_yaw_ -= 2 * M_PI;
        }
        enemy_color_ = (processed_msg.color == 0) ? "RED" : "BLUE";
        if (enemy_color_ == "RED") {
            params_.enemy_color = Params::RED;
        } else if (enemy_color_ == "BLUE") {
            params_.enemy_color = Params::BLUE;
        }
        if (light_detector_) {
            light_detector_->setEnemyColor(processed_msg.color == 0 ? Params::RED : Params::BLUE);
        }

        if (current_yaw_ < -M_PI/2 && last_yaw_rad_mcu_ > M_PI/2) {
            current_yaw_circle_mcu_ += 1;
        } else if (current_yaw_ > M_PI/2 && last_yaw_rad_mcu_ < -M_PI/2) {
            current_yaw_circle_mcu_ -= 1;
        }

        total_yaw_rad_mcu_ = current_yaw_circle_mcu_ * 2 * M_PI + current_yaw_;
        last_pitch_rad_mcu_ = current_pitch_;
        last_yaw_rad_mcu_ = current_yaw_;

        RCLCPP_DEBUG(this->get_logger(), 
            "Received serial data: v=%.2f, pitch=%.2f, yaw=%.2f, color=%s \nyaw_circle=%d, total_yaw_rad=%.2f",
            bullet_velocity_, current_pitch_, current_yaw_, enemy_color_.c_str(),
            current_yaw_circle_mcu_, total_yaw_rad_mcu_);


        if (!headIMUInfos.use_head_imu) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            DelayInfos now_serial_infos;
            now_serial_infos.last_pitch_rad_ = last_pitch_rad_mcu_;
            now_serial_infos.last_yaw_rad_ = last_yaw_rad_mcu_;
            now_serial_infos.last_roll_rad_ = 0.0;
            now_serial_infos.total_yaw_rad_ = total_yaw_rad_mcu_;
            now_serial_infos.push_time = current_time;
            serial_infos_delay_.push(now_serial_infos);
        }
    }

    void drawResults(cv::Mat& image, 
                     const std::vector<Light>& lights,
                     const std::vector<Armor>& armors,
                     const std::vector<ArmorResult>& classifyResults,
                     const PredictorResult& predictor_result) {
        // cv::Mat result = image.clone();
        cv::Mat& result = image;

        // 0. 绘制平面地面系不动点（DEBUG）
        cv::circle(result, ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);
        /* cv::circle(result, cv::Point2f(1000, 1000) - ground_stable_point, 10, cv::Scalar(0, 255, 0), 2);
        for (const auto& res : classifyResults) {
            for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                cv::line(result, res.corners[i] - ground_stable_point + cv::Point2f(500, 500), 
                        res.corners[(i+1)%4] - ground_stable_point + cv::Point2f(500, 500), 
                        cv::Scalar(0, 255, 0), 2);
            }    
        } */                       
        // 绘制3D面系不动点
        cv::Point3f test_point_pos = rest_frame_ -> worldToPnpP3f({0, 1000, 0});
        cv::Point2f test_point_pos_pixel = armor_solver_ -> project3DToPixel(test_point_pos);
        cv::circle(result, test_point_pos_pixel, 8, cv::Scalar(255, 0, 255), 2);

        // 1. 绘制灯条（绿色）
        for (const auto& light : lights) {
            cv::Point2f vertices[4];
            light.el.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(result, vertices[i], vertices[(i + 1) % 4], 
                        cv::Scalar(0, 255, 0), 2);
            }
        }

        // 2. 绘制装甲板候选区域（黄色）
        for (const auto& armor : armors) {
            for (size_t i = 0; i < armor.corners.size() && i < 4; i++) {
                cv::line(result, armor.corners[i], 
                        armor.corners[(i+1)%4], 
                        cv::Scalar(0, 255, 255), 2);
            }

            // 显示装甲板置信度
            if (!armor.corners.empty()) {
                std::string conf_str = cv::format("conf: %.2f", armor.confidence);
                cv::Point text_pos(armor.corners[0].x, armor.corners[0].y - 10);
                cv::putText(result, conf_str, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                        cv::Scalar(0, 255, 255), 1);
            }

            // 绘制灯条顶点
            for (size_t i = 0; i < armor.light_bar_corners.size() && i < 4; i++) {
                cv::line(result, armor.light_bar_corners[i], 
                        armor.light_bar_corners[(i+1)%4], 
                        cv::Scalar(255, 0, 0), 2);
            }
        }

        // 3. 绘制最终识别结果（红色）和跟踪信息
        for (const auto& res : classifyResults) {
            // 绘制装甲板轮廓
            if (res.is_tracked_now) {
                for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                    cv::line(result, res.corners[i], 
                            res.corners[(i+1)%4], 
                            cv::Scalar(0, 0, 255), 2);
                }    
            } else {
                for (size_t i = 0; i < res.corners.size() && i < 4; i++) {
                    cv::line(result, res.corners[i], 
                            res.corners[(i+1)%4], 
                            cv::Scalar(255, 0, 255), 2);
                }    
            }

            // 绘制灯条顶点
            for (size_t i = 0; i < res.armor.light_bar_corners.size() && i < 4; i++) {
                cv::line(result, res.armor.light_bar_corners[i], 
                        res.armor.light_bar_corners[(i+1)%4], 
                        cv::Scalar(0, 255, 255), 2);
            }

            // 绘制预测中心点
            for (auto& prediction : res.predictions) {
                cv::circle(result, prediction, 3, cv::Scalar(255, 0, 255), -1);
            }
            cv::circle(result, res.center_predicted, 3, cv::Scalar(0, 255, 255), -1);

            // 绘制中心点和编号
            cv::Point2f center = res.center;
            cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);

            const double distance_m = res.solve_armor_result.distance / 1000.0;
            std::string text = cv::format("N%d C%.2f T%d D%.2fm",
                                          res.number,
                                          res.confidence,
                                          res.is_tracked_now ? 1 : 0,
                                          distance_m);
            cv::Point text_pos(res.corners[1].x, res.corners[1].y - 10);

            // 使用黑色描边使文字更清晰
            cv::putText(result, text, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                        cv::Scalar(0, 0, 0), 3);
            cv::putText(result, text, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                        cv::Scalar(0, 0, 255), 1);

            const TargetManagerStatus& target_status =
                predictor_main_->targetManagerStatus();
            const bool is_target_candidate =
                target_status.target_type.has_value() &&
                res.number == static_cast<int>(*target_status.target_type);
            const bool is_measurement =
                predictor_result.has_measurement &&
                res.number == predictor_result.measurement_number &&
                cv::norm(res.center - predictor_result.measurement_center) < 0.5;

            if (is_target_candidate) {
                cv::putText(result, "TARGET CANDIDATE",
                            cv::Point(center.x - 45, center.y + 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
            if (is_measurement) {
                cv::putText(result, "MEASUREMENT",
                            cv::Point(center.x - 45, center.y + 52),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

//         // 在窗口中显示图像
// #ifdef SHOW_WINDOWS
//         cv::imshow("Armor Detection", result);
//         cv::waitKey(1);  // 确保窗口刷新
// #endif
    }

    void writeYawDebugCsv(std::uint64_t frame_id,
                          double timestamp_s,
                          const PredictorResult& predictor_result,
                          const TargetManagerStatus& target_status) {
        if (!yaw_debug_csv_enabled_ || !yaw_debug_csv_ ||
            !predictor_result.yaw_debug.available) {
            return;
        }

        const YawMeasurementDebug& debug = predictor_result.yaw_debug;
        std::string target_name = "NONE";
        if (target_status.target_type.has_value()) {
            const auto index =
                static_cast<std::size_t>(*target_status.target_type);
            if (index < ArmorType::ArmorTypeStrings.size()) {
                target_name = ArmorType::ArmorTypeStrings[index];
            }
        }

        yaw_debug_csv_ << frame_id << ',' << std::setprecision(12)
                       << timestamp_s << ',' << target_name << ','
                       << debug.measurement_number << ','
                       << debug.measurement_world_mm.x << ','
                       << debug.measurement_world_mm.y << ','
                       << debug.measurement_world_mm.z << ','
                       << debug.yaw_raw_rad << ','
                       << debug.yaw_refined_rad << ','
                       << debug.yaw_used_rad << ','
                       << debug.yaw_delta_rad << ','
                       << debug.reprojection_rmse_raw_px << ','
                       << debug.reprojection_rmse_refined_px << ','
                       << debug.facing_angle_rad << ','
                       << (debug.refined_valid ? 1 : 0) << ','
                       << debug.refinement_status << ','
                       << debug.ekf_yaw_rad << ','
                       << debug.ekf_w_rad_s << ','
                       << debug.nis << ','
                       << debug.ekf_state << ','
                       << debug.target_state << ','
                       << debug.matched_armor_id << ','
                       << (debug.armor_switched ? 1 : 0) << '\n';
    }

    void writePnpDebugCsv(
        std::uint64_t frame_id,
        double timestamp_s,
        std::size_t detection_index,
        const ArmorResult& armor_result,
        const AimResult& solve_result,
        int frame_width,
        int frame_height) {
        if (!pnp_debug_csv_enabled_ || !pnp_debug_csv_ ||
            !solve_result.pnp_diagnostic.available) {
            return;
        }
        const PnPDiagnostic& diagnostic = solve_result.pnp_diagnostic;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        pnp_debug_csv_ << frame_id << ',' << std::setprecision(12)
                       << timestamp_s << ',' << detection_index << ','
                       << armor_result.number << ','
                       << (armor_result.is_large ? 1 : 0) << ','
                       << (armor_result.is_tracked_now ? 1 : 0) << ','
                       << frame_width << ',' << frame_height << ','
                       << (calibration_resolution_known_ ? 1 : 0) << ','
                       << calibration_width_ << ',' << calibration_height_ << ','
                       << detector_input_width_ << ',' << detector_input_height_ << ','
                       << static_cast<double>(frame_width) /
                              std::max(1, detector_input_width_) << ','
                       << static_cast<double>(frame_height) /
                              std::max(1, detector_input_height_) << ','
                       << diagnostic.current_corner_length_scale << ','
                       << diagnostic.object_width_mm << ','
                       << diagnostic.object_height_mm;

        const auto write_corners = [this, nan](
            const std::vector<cv::Point2f>& corners) {
            for (int index = 0; index < 4; ++index) {
                if (index < static_cast<int>(corners.size())) {
                    pnp_debug_csv_ << ',' << corners[index].x
                                   << ',' << corners[index].y;
                } else {
                    pnp_debug_csv_ << ',' << nan << ',' << nan;
                }
            }
        };
        write_corners(armor_result.armor.raw_detector_corners);
        write_corners(armor_result.armor.light_bar_corners);

        const auto write_pose = [this, nan](const PnPDiagnosticPose& pose) {
            cv::Point3f world_position(
                static_cast<float>(nan),
                static_cast<float>(nan),
                static_cast<float>(nan));
            if (pose.valid) {
                world_position =
                    rest_frame_->pnpToWorldP3f(pose.muzzle_position_mm);
            }
            pnp_debug_csv_
                << ',' << (pose.valid ? 1 : 0)
                << ',' << (pose.valid ? pose.camera_position_mm.x / 1000.0 : nan)
                << ',' << (pose.valid ? pose.camera_position_mm.y / 1000.0 : nan)
                << ',' << (pose.valid ? pose.camera_position_mm.z / 1000.0 : nan)
                << ',' << (pose.valid ? world_position.x / 1000.0 : nan)
                << ',' << (pose.valid ? world_position.y / 1000.0 : nan)
                << ',' << (pose.valid ? world_position.z / 1000.0 : nan)
                << ',' << (pose.valid ? pose.range_mm / 1000.0 : nan)
                << ',' << (pose.valid ? pose.reprojection_rmse_px : nan)
                << ',' << (pose.valid ? pose.world_yaw_rad : nan)
                << ',' << (pose.valid ? pose.facing_angle_rad : nan);
        };
        write_pose(diagnostic.raw_detector_corners);
        write_pose(diagnostic.current_corrected_corners);
        for (const PnPScaleDiagnostic& scale : diagnostic.object_scale_sweep) {
            write_pose(scale.pose);
        }
        pnp_debug_csv_ << '\n';
    }

    void writeGeometryDebugCsv(
        std::uint64_t frame_id,
        double timestamp_s,
        const PredictorResult& predictor_result,
        const TargetManagerStatus& target_status) {
        if (!geometry_debug_csv_enabled_ || !geometry_debug_csv_ ||
            !predictor_result.geometry_debug.available) {
            return;
        }

        const GeometryDebug& debug = predictor_result.geometry_debug;
        std::string target_name = "NONE";
        if (target_status.target_type.has_value()) {
            const auto index =
                static_cast<std::size_t>(*target_status.target_type);
            if (index < ArmorType::ArmorTypeStrings.size()) {
                target_name = ArmorType::ArmorTypeStrings[index];
            }
        }

        geometry_debug_csv_ << frame_id << ',' << std::setprecision(12)
                            << timestamp_s << ',' << target_name << ','
                            << debug.measurement_number << ','
                            << (predictor_result.has_measurement ? 1 : 0) << ','
                            << debug.target_state << ',' << debug.ekf_state << ','
                            << (debug.updated ? 1 : 0) << ','
                            << (debug.measurement_valid ? 1 : 0) << ','
                            << debug.current_armor_id << ','
                            << debug.r1_m << ',' << debug.r2_m << ','
                            << debug.h_m << ',' << debug.p_r1_m2 << ','
                            << debug.p_r2_m2 << ',' << debug.p_h_m2 << ','
                            << debug.center_x_m << ',' << debug.center_y_m << ','
                            << debug.center_z_m << ',' << debug.state_yaw_rad << ','
                            << debug.w_rad_s << ',' << debug.nis << ','
                            << debug.matched_armor_id << ','
                            << debug.armor_parity << ','
                            << (debug.armor_switched ? 1 : 0) << ','
                            << (debug.direction_reversal ? 1 : 0) << ','
                            << (debug.pending_sign_conflict ? 1 : 0) << ','
                            << (debug.recovered ? 1 : 0) << ','
                            << (debug.geometry_valid ? 1 : 0) << ','
                            << (debug.geometry_update_allowed ? 1 : 0) << ','
                            << (debug.geometry_preserved ? 1 : 0) << '\n';
    }

    void writeAssociationDebugCsv(
        std::uint64_t frame_id,
        double timestamp_s,
        const PredictorResult& predictor_result,
        const TargetManagerStatus& target_status) {
        if (!association_debug_csv_enabled_ || !association_debug_csv_ ||
            !predictor_result.geometry_debug.available ||
            !predictor_result.has_measurement) {
            return;
        }

        const GeometryDebug& debug = predictor_result.geometry_debug;
        std::string target_name = "NONE";
        if (target_status.target_type.has_value()) {
            const auto index =
                static_cast<std::size_t>(*target_status.target_type);
            if (index < ArmorType::ArmorTypeStrings.size()) {
                target_name = ArmorType::ArmorTypeStrings[index];
            }
        }

        for (const rm_ekf::AssociationHypothesisDebug& hypothesis :
             debug.association_hypotheses) {
            const rm_ekf::AssociationCandidate& measurement =
                hypothesis.measurement;
            const rm_ekf::AssociationCandidate& hypothetical =
                hypothesis.hypothetical_scaled_yaw_measurement;
            const rm_ekf::AssociationCandidate& fixed_radius_covariance =
                hypothesis.statistically_fixed_radius_measurement;
            association_debug_csv_ << frame_id << ',' << std::setprecision(12)
                << timestamp_s << ',' << target_name << ','
                << debug.target_state << ',' << debug.ekf_state << ','
                << debug.tracker_state_before << ','
                << debug.measurement_number << ',' << debug.current_armor_id << ','
                << debug.best_id << ','
                << (debug.candidate_is_switch ? 1 : 0) << ','
                << (debug.armor_switched ? 1 : 0) << ','
                << (debug.temp_lost_recovery ? 1 : 0) << ','
                << (debug.recovered ? 1 : 0) << ','
                << (debug.phase_observer_valid ? 1 : 0) << ','
                << debug.phase_delta << ',' << debug.phase_w_filtered << ','
                << (debug.pending_sign_conflict ? 1 : 0) << ','
                << (debug.direction_reversal ? 1 : 0) << ','
                << (debug.topology_event ? 1 : 0) << ','
                << debug.measurement_yaw << ',' << debug.predicted_yaw << ','
                << debug.yaw_innovation << ','
                << debug.hypothetical_scaled_nis << ','
                << debug.hypothetical_scaled_nis_contribution(0) << ','
                << debug.hypothetical_scaled_nis_contribution(1) << ','
                << debug.hypothetical_scaled_nis_contribution(2) << ','
                << debug.hypothetical_scaled_nis_contribution(3) << ','
                << hypothesis.armor_id << ','
                << (hypothesis.armor_id == debug.current_armor_id ? 1 : 0) << ','
                << (debug.updated &&
                    hypothesis.armor_id == debug.matched_armor_id ? 1 : 0) << ','
                << debug.center_x_m << ',' << debug.center_y_m << ','
                << debug.center_z_m << ',' << debug.state_yaw_rad << ','
                << debug.r1_m << ',' << debug.r2_m << ',' << debug.h_m << ','
                << debug.p_r1_m2 << ',' << debug.p_r2_m2 << ','
                << debug.p_h_m2 << ',' << hypothesis.predicted.x << ','
                << hypothesis.predicted.y << ',' << hypothesis.predicted.z << ','
                << hypothesis.predicted.yaw << ',' << hypothesis.facing_angle << ','
                << (hypothesis.range_pass ? 1 : 0) << ','
                << (hypothesis.visibility_pass ? 1 : 0) << ','
                << measurement.innovation(0) << ','
                << measurement.innovation(1) << ','
                << measurement.innovation(2) << ','
                << measurement.innovation(3) << ','
                << measurement.position_error << ',' << measurement.yaw_error << ','
                << measurement.yaw_variance_scale << ','
                << measurement.nis << ',' << measurement.nis_contribution(0) << ','
                << measurement.nis_contribution(1) << ','
                << measurement.nis_contribution(2) << ','
                << measurement.nis_contribution(3) << ','
                << hypothetical.nis << ','
                << hypothetical.nis_contribution(0) << ','
                << hypothetical.nis_contribution(1) << ','
                << hypothetical.nis_contribution(2) << ','
                << hypothetical.nis_contribution(3) << ','
                << fixed_radius_covariance.nis << ','
                << fixed_radius_covariance.nis_contribution(0) << ','
                << fixed_radius_covariance.nis_contribution(1) << ','
                << fixed_radius_covariance.nis_contribution(2) << ','
                << fixed_radius_covariance.nis_contribution(3) << ','
                << (hypothesis.nis_gate_pass ? 1 : 0) << ','
                << (hypothesis.position_gate_pass ? 1 : 0) << ','
                << (hypothesis.yaw_gate_pass ? 1 : 0) << ','
                << (hypothesis.passes_all_measurement_gates ? 1 : 0) << ','
                << hypothesis.radial_residual << ','
                << hypothesis.tangential_residual << ','
                << (debug.updated ? 1 : 0) << '\n';
        }
    }

    void writeLifecycleDebugCsv(
        std::uint64_t frame_id,
        double timestamp_s,
        const PredictorResult& predictor_result,
        const TargetManagerStatus& target_status) {
        if (!lifecycle_debug_csv_enabled_ || !lifecycle_debug_csv_) return;

        std::string target_name = "NONE";
        if (target_status.target_type.has_value()) {
            const auto index =
                static_cast<std::size_t>(*target_status.target_type);
            if (index < ArmorType::ArmorTypeStrings.size()) {
                target_name = ArmorType::ArmorTypeStrings[index];
            }
        }
        const GeometryDebug& debug = predictor_result.geometry_debug;
        lifecycle_debug_csv_ << frame_id << ',' << std::setprecision(12)
            << timestamp_s << ',' << TargetManager::stateName(target_status.state)
            << ',' << target_name << ','
            << (predictor_result.has_measurement ? 1 : 0) << ','
            << predictor_result.measurement_number << ','
            << (debug.available ? debug.ekf_state : "NONE") << ','
            << (debug.available && debug.updated ? 1 : 0) << ','
            << (debug.available && debug.armor_switched ? 1 : 0) << ','
            << (debug.available ? debug.nis
                                : std::numeric_limits<double>::quiet_NaN())
            << '\n';
    }

    void updateEKFDebugPlot(
        std::uint64_t frame_id,
        double timestamp_s,
        const PredictorResult& predictor_result) {
        if (!ekf_debug_plotter_ || !ekf_debug_plotter_->active() ||
            !predictor_result.geometry_debug.available) {
            return;
        }
        const GeometryDebug& d = predictor_result.geometry_debug;
        EKFDebugPlotSample s;
        s.frame_id = frame_id;
        s.timestamp_s = timestamp_s;
        s.measurement_x = d.measurement(0);
        s.measurement_y = d.measurement(1);
        s.measurement_z = d.measurement(2);
        s.measurement_yaw = d.measurement(3);
        s.matched_id = d.matched_armor_id;
        s.pre_pred_x = d.pre_predicted(0);
        s.pre_pred_y = d.pre_predicted(1);
        s.pre_pred_z = d.pre_predicted(2);
        s.pre_pred_yaw = d.pre_predicted(3);
        s.post_pred_x = d.post_predicted(0);
        s.post_pred_y = d.post_predicted(1);
        s.post_pred_z = d.post_predicted(2);
        s.post_pred_yaw = d.post_predicted(3);
        s.pre_dx = d.pre_residual(0);
        s.pre_dy = d.pre_residual(1);
        s.pre_dz = d.pre_residual(2);
        s.pre_position_error = d.pre_position_error;
        s.post_dx = d.post_residual(0);
        s.post_dy = d.post_residual(1);
        s.post_dz = d.post_residual(2);
        s.post_position_error = d.post_position_error;
        s.residual_radial = d.residual_radial;
        s.residual_tangential = d.residual_tangential;
        s.center_x = d.center_x_m;
        s.center_y = d.center_y_m;
        s.center_z = d.center_z_m;
        s.vx = d.vx_m_s;
        s.vy = d.vy_m_s;
        s.vz = d.vz_m_s;
        s.yaw = d.state_yaw_rad;
        s.w = d.w_rad_s;
        s.phase_w = d.phase_w_filtered;
        s.instant_phase_w = d.phase_w_instant;
        s.phase_valid = d.phase_observer_valid;
        if (!s.phase_valid) s.instant_phase_w =
            std::numeric_limits<double>::quiet_NaN();
        s.r1 = d.r1_m;
        s.r2 = d.r2_m;
        s.h = d.h_m;
        s.p_x = d.p_x_m2;
        s.p_vx = d.p_vx_m2_s2;
        s.p_y = d.p_y_m2;
        s.p_vy = d.p_vy_m2_s2;
        s.p_r1 = d.p_r1_m2;
        s.p_r2 = d.p_r2_m2;
        s.p_h = d.p_h_m2;
        s.nis = d.nis;
        s.nis_xyz = d.nis_xyz;
        s.nis_yaw = d.nis_yaw;
        s.yaw_variance_scale = d.yaw_variance_scale;
        s.tracker_state = d.ekf_state;
        s.armor_switch = d.armor_switched;
        s.direction_reversal = d.direction_reversal;
        s.pending_sign_conflict = d.pending_sign_conflict;
        s.association_success = d.updated;
        ekf_debug_plotter_->update(std::move(s));

        if (ekf_debug_plotter_->windowEnabled()) {
            cv::Mat curves = ekf_debug_plotter_->render();
            cv::imshow("EKF Debug Curves", curves);
            if (ekf_debug_plotter_->saveLatestPng() && frame_id % 30 == 0) {
                cv::imwrite(ekf_debug_plotter_->screenshotPath(), curves);
            }
            cv::waitKey(1);
        }
    }

    void processImage() {
    

        if (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - headIMUInfos.last_mcu_yaw_update_time).count() > 3000
        ) {
            if (fabs(headIMUInfos.last_mcu_command_yaw - headIMUInfos.latest_mcu_command_yaw_when_mcu_yaw_update)
                > 5.0 * M_PI / 180.0
            ) {
                headIMUInfos.mcu_yaw_online = false;
            }
        }



        while (serial_infos_delay_.size() > 1 && 
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - serial_infos_delay_.front().push_time).count() > serial_delay_time) {
            serial_infos_delay_.pop();
        }
        DelayInfos delayed_serial_infos = serial_infos_delay_.front();
        last_pitch_rad_delayed_ = delayed_serial_infos.last_pitch_rad_;
        last_yaw_rad_delayed_ = delayed_serial_infos.last_yaw_rad_;
        total_yaw_rad_delayed_ = delayed_serial_infos.total_yaw_rad_;
        last_roll_rad_delayed_ = delayed_serial_infos.last_roll_rad_;
        ground_stable_point = cv::Point2f(500+total_yaw_rad_delayed_*yaw_rad_to_x_pixel_ratio, 500+last_pitch_rad_delayed_*pitch_rad_to_y_pixel_ratio);
        rest_frame_ -> updateCamOrientation(last_yaw_rad_delayed_, last_pitch_rad_delayed_, last_roll_rad_delayed_);
        rest_frame_ -> updateCamPosition(0, 0, 0); // 预留位置接口
        predictor_main_ -> update_serial_info(bullet_velocity_, last_pitch_rad_delayed_, last_yaw_rad_delayed_, total_yaw_rad_delayed_);
        RCLCPP_DEBUG(this->get_logger(), "ground_stable_point: %.2f %.2f", ground_stable_point.x, ground_stable_point.y);




        
        cv::Mat frame;
        double frame_timestamp_s = 0.0;
        [[maybe_unused]] std::uint64_t frame_id = 0;
#if defined(USE_VIDEO) || defined(USE_IMAGES) || defined(SYNC_CAMERA_FPS)
        while (image_used)
        {
            usleep(1000);
        }
#endif
        pthread_mutex_lock(&g_mutex);
        if (!g_frame_packet.image.empty()) {
            frame = g_frame_packet.image.clone();
            frame_timestamp_s = g_frame_packet.timestamp_s;
            frame_id = g_frame_packet.frame_id;
            image_used = true;
        }
        pthread_mutex_unlock(&g_mutex);

        if (!frame.empty()) {
#ifdef SAVE_IMG_FREQ
            frame_count_ += 1;
            if (frame_count_ % SAVE_IMG_FREQ == 0 && frame_count_ / SAVE_IMG_FREQ < 2000) {
                // 创建保存目录
                fs::create_directories("camera_images");
                // 生成文件名（00001.jpg 格式）
                std::ostringstream filename;
                filename << "camera_images/"
                        << std::setw(5) << std::setfill('0') << (frame_count_ / SAVE_IMG_FREQ)
                        << ".jpg";
                cv::imwrite(filename.str(), frame);
            }
#endif

#if (defined LOG_RESULT_VIDEO) or (defined LOG_ORIGIN_VIDEO)
            two_video_logger -> updateOriginFrame(frame);
#endif

            //cv::resize(frame, frame, cv::Size(768, 512), 0, 0, cv::INTER_LINEAR);

            //cv::flip(frame, frame, -1);  // 翻转图像（上下翻转）

            std::vector<Light> lights;
            std::vector<Armor> armors;
            std::vector<ArmorResult> classifyResults;

            if (use_RP24_YOLO) {
                // armors = rp24_yolo_wrapper -> detectArmors(frame, enemy_color_);
                // for (Armor& armor : armors) {
                //     lights.emplace_back(armor.leftLight);
                //     lights.emplace_back(armor.rightLight);
                // }
                // classifyResults = classifier_->classify(frame, armors, ground_stable_point);
                classifyResults = rp24_yolo_wrapper -> detectArmorsWithClassifyAndTrack(frame, enemy_color_, ground_stable_point, &armors);
                for (Armor& armor : armors) {
                    lights.emplace_back(armor.leftLight);
                    lights.emplace_back(armor.rightLight);
                }
            } else {

                // 检测灯条
                light_detector_->detectLights(frame);
                light_detector_->processLights();
                lights = light_detector_->getLights();
                
                // 检测装甲板
                armors = armor_detector_->detectArmors(lights);
                classifyResults = classifier_->classify(frame, armors, ground_stable_point);
            }

            std::vector<ArmorResult> classifyResults_withSolveArmorResult;
            std::size_t pnp_detection_index = 0;
            for (ArmorResult &classify_result : classifyResults) {
                AimResult solve_armor_result = armor_solver_->solveArmor(
                    classify_result, last_pitch_rad_delayed_,
                    last_yaw_rad_delayed_, last_roll_rad_delayed_);
                writePnpDebugCsv(
                    frame_id, frame_timestamp_s, pnp_detection_index,
                    classify_result, solve_armor_result,
                    frame.cols, frame.rows);
                ++pnp_detection_index;
                cv::Point3f rest_frame_pos = rest_frame_ -> pnpToWorldP3f(solve_armor_result.position);
                if (rest_frame_pos.z < max_armor_position_height && solve_armor_result.valid) { // 高度高于一定值视为无效
                    classifyResults_withSolveArmorResult.emplace_back(classify_result);
                    classifyResults_withSolveArmorResult.back().solve_armor_result = solve_armor_result;
                }
            }

            bool auto_aim_switch = true;
            if (((!headIMUInfos.last_auto_aim_switch) && auto_aim_switch) &&
                (headIMUInfos.use_head_imu && (!headIMUInfos.mcu_yaw_online))
            ) { // 仅在使用IMU且电控yaw轴数据掉线时在开启自瞄时校准
                recalibrateHeadIMU();
            }
            headIMUInfos.last_auto_aim_switch = auto_aim_switch;
            PredictorResult predictor_result = predictor_main_ -> step(
                                                                       classifyResults_withSolveArmorResult,
                                                                       frame,
                                                                       frame_timestamp_s,
                                                                       ArmorType::Nearest,
                                                                       auto_aim_switch, headIMUInfos.mcu_yaw_online); // Todo
            float mcu_command_pitch = predictor_result.command_pitch;
            float mcu_command_yaw = predictor_result.command_yaw;
            if (headIMUInfos.use_head_imu) {
                mcu_command_pitch = predictor_result.command_pitch; // + headIMUInfos.to_mcu_delta_pitch;
                mcu_command_yaw = predictor_result.command_yaw + headIMUInfos.to_mcu_delta_yaw;
            }
            headIMUInfos.last_mcu_command_yaw = mcu_command_yaw;
            if (predictor_result.reset) {
                // RCLCPP_INFO(this->get_logger(), "send data: yaw[%.2f] pitch[%.2f] fire[%d]", 0.0, 0.0, false);
                serial_communication_->sendData(0.0, 0.0, false);
            } else {
                // RCLCPP_INFO(this->get_logger(), "send data: yaw[%.2f] pitch[%.2f] fire[%d]", predictor_result.command_pitch, predictor_result.command_yaw, predictor_result.fire_flag);
                serial_communication_->sendData(mcu_command_pitch, mcu_command_yaw, predictor_result.fire_flag);
            }
            
            // 显示当前参数状态
            cv::putText(frame, 
                cv::format("V: %.1f m/s, P: %.1f, Y: %.1f", 
                    bullet_velocity_, last_pitch_rad_delayed_, last_yaw_rad_delayed_),
                cv::Point(20, 195),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(frame, 
                "enemy_color: " + enemy_color_, 
                cv::Point2f(20,225),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            const TargetManagerStatus& target_status =
                predictor_main_->targetManagerStatus();
            writeYawDebugCsv(frame_id, frame_timestamp_s,
                             predictor_result, target_status);
            writeGeometryDebugCsv(frame_id, frame_timestamp_s,
                                  predictor_result, target_status);
            writeAssociationDebugCsv(frame_id, frame_timestamp_s,
                                     predictor_result, target_status);
            writeLifecycleDebugCsv(frame_id, frame_timestamp_s,
                                   predictor_result, target_status);
            updateEKFDebugPlot(frame_id, frame_timestamp_s, predictor_result);
            const std::string aiming_text = target_status.target_type.has_value()
                ? "aiming " + ArmorType::ArmorTypeStrings[*target_status.target_type] +
                    ": " + ((*target_status.target_type == ArmorType::Outpost ||
                              *target_status.target_type == ArmorType::Base)
                                 ? "DIRECT"
                                 : "EKF")
                : "aiming NONE";
            cv::putText(frame, aiming_text,
                cv::Point2f(20, 255),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);

            drawRestFrame(frame, rest_frame_, armor_solver_);

            drawResults(frame, lights, armors,
                        classifyResults_withSolveArmorResult, predictor_result);

            yaw_visualizer_ -> update(last_yaw_rad_delayed_ + (headIMUInfos.use_head_imu ? headIMUInfos.to_mcu_delta_yaw : 0.0), mcu_command_yaw);
            // yaw_visualizer_ -> show();
            cv::Mat yaw_visualizer_frame = yaw_visualizer_ -> getDisplay();

            //计算帧率
            fps_counter->tick();

            cv::putText(frame, 
                cv::format("frame rate: %.1f fps", fps_counter->fps()), 
                cv::Point(20, 285),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            cv::putText(frame, 
                cv::format("since start: %.4f s", static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - node_start_time).count()) / 1000.0f), 
                cv::Point(20, 315),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);
            auto system_clock_now = std::chrono::system_clock::now();
            std::time_t system_clock_now_t = std::chrono::system_clock::to_time_t(system_clock_now);
            std::tm* system_clock_now_tm = std::localtime(&system_clock_now_t);
            char system_clock_now_str_buffer[80];
            std::strftime(system_clock_now_str_buffer, sizeof(system_clock_now_str_buffer), "%Y-%m-%d %H:%M:%S", system_clock_now_tm);
            cv::putText(frame, 
                cv::format("system_clock: %s", system_clock_now_str_buffer), 
                cv::Point(20, 345),
                cv::FONT_HERSHEY_COMPLEX, 0.7, 
                cv::Scalar(0, 255, 0), 1, 8, false);

#if (defined LOG_RESULT_VIDEO) or (defined LOG_ORIGIN_VIDEO)
            two_video_logger -> updateDrewFrame(frame);
            two_video_logger -> updateRMMFrame(predictor_result.info_images.RMM_visualize_frame);
            two_video_logger -> updateCDOFrame(predictor_result.info_images.common_debug_oscilloscope_frame);
            two_video_logger -> updateYawFrame(yaw_visualizer_frame);
            two_video_logger -> updateComFrame(com_data_visualize_frame);
            com_data_visualize_frame_used = true;
            two_video_logger -> writeTwoFrame();
#endif
                
#ifdef SHOW_WINDOWS
            // EKF-only debug mode: mirror the standalone v4 replay windows.
            if (!predictor_result.info_images.RMM_visualize_frame.empty()) {
                cv::imshow("armor_ekf_real_project",
                           predictor_result.info_images.RMM_visualize_frame);
            }
            if (!predictor_result.info_images.EKF_vertical_frame.empty()) {
                cv::imshow("armor_ekf_real_project_vertical",
                           predictor_result.info_images.EKF_vertical_frame);
            }
            if (!predictor_result.info_images.EKF_camera_overlay_frame.empty()) {
                cv::imshow("armor_ekf_camera_overlay",
                           predictor_result.info_images.EKF_camera_overlay_frame);
            }
            cv::waitKey(1);
#endif


            if (std::chrono::steady_clock::now() - last_feed_dog_time >= std::chrono::seconds(3)) {
                watchdog_client -> feed();
                last_feed_dog_time = std::chrono::steady_clock::now();
            } // 正常运行时，每3秒喂一次狗
        }

        // 获取处理帧率
        RCLCPP_INFO(this->get_logger(), "frame rate: %.1f fps\n" , fps_counter->fps());
    }

    // 参数文件
    std::shared_ptr<YAML::Node> config_file_ptr; 

    // 成员变量
    // rclcpp::TimerBase::SharedPtr timer_;
    std::thread com_timer_thread_;
    std::thread main_loop_thread_;
    
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<LightBarDetector> light_detector_;
    std::shared_ptr<ArmorDetector> armor_detector_;
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<ArmorClassifier> classifier_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;

    std::shared_ptr<VideoInput> video_input_;
    std::shared_ptr<ImagesInput> images_input_;
    float frame_rate_;

    std::shared_ptr<RestFrame> rest_frame_;

    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    
    float bullet_velocity_;


    float last_pitch_rad_mcu_;
    float last_yaw_rad_mcu_;
    float total_yaw_rad_mcu_;
    int current_yaw_circle_mcu_ = 0;
    
    float last_pitch_rad_imu_;
    float last_yaw_rad_imu_;
    float total_yaw_rad_imu_;
    float last_roll_rad_imu_;
    int current_yaw_circle_imu_ = 0;

    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;
    float last_roll_rad_delayed_ = 0;
    struct DelayInfos {
        float last_pitch_rad_;
        float last_yaw_rad_;
        float last_roll_rad_;
        float total_yaw_rad_;
        std::chrono::steady_clock::time_point push_time;
    };
    std::queue<DelayInfos> serial_infos_delay_;
    float serial_delay_time;
    std::string enemy_color_;
    Params params_;

    // 帧率计算器
    std::shared_ptr<FrameRateCounter> fps_counter;
#ifdef SAVE_IMG_FREQ
    long long frame_count_ = 0;
#endif
    cv::Point2f ground_stable_point;
    std::shared_ptr<SerialCommunicationClass> serial_communication_;
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;

    std::shared_ptr<PredictorMain> predictor_main_;

    std::shared_ptr<RP24YOLOWrapper> rp24_yolo_wrapper;
    bool use_RP24_YOLO;

    float max_armor_position_height;

    std::shared_ptr<WatchdogClient> watchdog_client;
    std::chrono::steady_clock::time_point last_feed_dog_time;

    struct {
        std::shared_ptr<HeadIMUSerialCommunicationClass> headIMU_communication_;
        std::thread headIMU_timer_thread_;

        bool use_head_imu = true;

        float head_imu_yaw;
        float head_imu_pitch;
        float head_imu_roll;

        float mcu_yaw;
        float mcu_pitch;
        
        float last_mcu_yaw;
        float latest_head_imu_yaw_when_mcu_yaw_update;
        std::chrono::steady_clock::time_point last_mcu_yaw_update_time;
        bool mcu_yaw_online = true;
        float last_mcu_command_yaw;
        float latest_mcu_command_yaw_when_mcu_yaw_update;

        float to_mcu_delta_yaw;
        float to_mcu_delta_pitch;

        bool last_auto_aim_switch = true; // 用于在开启自瞄时进行校准
    } headIMUInfos;

    std::shared_ptr<YawVisualizer> yaw_visualizer_;

    std::shared_ptr<TwoVideoLogger> two_video_logger;
    cv::Mat com_data_visualize_frame;
    bool com_data_visualize_frame_used = true;

    bool yaw_debug_csv_enabled_ = false;
    std::ofstream yaw_debug_csv_;
    bool pnp_debug_csv_enabled_ = false;
    std::ofstream pnp_debug_csv_;
    bool calibration_resolution_known_ = false;
    int calibration_width_ = 0;
    int calibration_height_ = 0;
    int detector_input_width_ = 640;
    int detector_input_height_ = 640;
    bool geometry_debug_csv_enabled_ = false;
    std::ofstream geometry_debug_csv_;
    bool association_debug_csv_enabled_ = false;
    std::ofstream association_debug_csv_;
    bool lifecycle_debug_csv_enabled_ = false;
    std::ofstream lifecycle_debug_csv_;
    std::unique_ptr<EKFDebugPlotter> ekf_debug_plotter_;
};

std::shared_ptr<ArmorDetectNode> node;
void signalHandler(int signum) {
    if (node) {
        rclcpp::shutdown();
    }
}
int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    node = std::make_shared<ArmorDetectNode>();
    signal(SIGINT, signalHandler);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
