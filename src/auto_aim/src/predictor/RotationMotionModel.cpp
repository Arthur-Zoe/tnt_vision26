// RotationMotionModel.cpp
#include "predictor/RotationMotionModel.h"

using namespace Eigen;

RotationMotionModel::RotationMotionModel(
    ObservedData& initObservedData,
    std::shared_ptr<RestFrame> rest_frame,
    bool outpost,
    double init_r,
    std::shared_ptr<YAML::Node> config_file_ptr)
    : last_observed_data(initObservedData),
      rest_frame_(rest_frame),
      is_outpost(outpost),
      config_file_ptr_(config_file_ptr) {
    last_update_time_ = initObservedData.t;
    center_vx = 0.0;
    center_vy = 0.0;
    center_vz = 0.0;

    // Normal 4-armor vehicles: use the validated v4 11D EKF/tracker.
    if (!is_outpost) {
        robust_ekf_active_ = true;
        n_armors = 4;
        jump_rad = M_PI / 2.0;
        rotation_direction = 1;
        max_history = 0;

        rm_ekf::RobustTrackerConfig robust_cfg;
        if (config_file_ptr_) {
            robust_cfg = rm_ekf::RobustTrackerConfig::fromYaml(
                *config_file_ptr_, init_r / 1000.0);
        } else {
            robust_cfg.initial_filter.r1 = init_r / 1000.0;
            robust_cfg.initial_filter.r2 = init_r / 1000.0;
            robust_cfg.initial_filter.h = 0.0;
        }

        robust_tracker_ = std::make_unique<rm_ekf::RobustArmorTracker>();
        robust_tracker_->configure(robust_cfg);

        rm_ekf::ArmorObservation obs;
        obs.x = initObservedData.x / 1000.0;
        obs.y = initObservedData.y / 1000.0;
        obs.z = initObservedData.z / 1000.0;
        obs.yaw = initObservedData.yaw;
        robust_last_result_ = robust_tracker_->process(obs, 0.0, -1);
        update_frames_count = 1;
        syncLegacyFieldsFromRobust();
        return;
    }

    // Outpost: retain the existing 3-armor model for now.
    observedDataHistory.push_back(initObservedData);
    angle_ekf_ = std::make_unique<AngleEKF>(true);

    n_armors = 3;
    r_now = 276.5;
    r_another = 276.5;
    r_now_prev_ = r_now;
    r_another_prev_ = r_another;
    regularization_weight_ = 1.0;

    center_x = initObservedData.x - r_now * sin(initObservedData.yaw);
    center_y = initObservedData.y + r_now * cos(initObservedData.yaw);
    center_z = initObservedData.z;
    z_another = center_z;

    max_history = 90;
    rotation_direction = 1;
    jump_rad = M_PI * 2.0 / n_armors;
    lambda_ = -std::log(0.97) / 0.033;
    resetExponentialLS();
}

void RotationMotionModel::syncLegacyFieldsFromRobust() {
    if (!robust_ekf_active_ || !robust_tracker_) {
        return;
    }

    const rm_ekf::ArmorState s = robust_tracker_->state();
    center_x = s.x * 1000.0;
    center_y = s.y * 1000.0;
    center_z = s.z * 1000.0;
    center_vx = s.vx * 1000.0;
    center_vy = s.vy * 1000.0;
    center_vz = s.vz * 1000.0;
    r_now = s.r1 * 1000.0;
    r_another = s.r2 * 1000.0;
    z_another = (s.z + s.h) * 1000.0;
    rotation_direction = (s.w >= 0.0) ? 1 : -1;

    const int matched = robust_tracker_->currentArmorId();
    if (matched >= 0) debug_flip_flag = (matched % 2) + 1;
}

void RotationMotionModel::resetExponentialLS() {
    // 初始化7x7协方差矩阵
    P_center_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1000.0;
    
    // 初始状态估计 [center_x, center_y, center_z, center_vx, center_vy, center_vz, r]
    x_center_ = Eigen::VectorXd::Zero(STATE_DIM);
    x_center_(0) = center_x;  // center_x
    x_center_(1) = center_y;  // center_y
    x_center_(2) = center_z;  // center_z (新增)
    x_center_(3) = 0.0;       // center_vx
    x_center_(4) = 0.0;       // center_vy
    x_center_(5) = 0.0;       // center_vz (新增)
    x_center_(6) = r_now;         // r
}

void RotationMotionModel::updateExponentialLS(double armor_x, double armor_y, double armor_z, double armor_yaw, double t, double dt, double weight, double delta_r, double delta_z) {
    // 构建测量方程
    
    double cosYaw = cos(armor_yaw);
    double sinYaw = sin(armor_yaw);
    double offAxisX = cosYaw;
    double offAxisY = sinYaw;
    
    if (is_outpost) {
        // 测量1的值：装甲板在垂直方向上的投影应为0
        double z1 = offAxisX * armor_x + offAxisY * armor_y;
        
        // 测量1的测量矩阵 - 只更新center_x, center_y
        Eigen::RowVectorXd H1(STATE_DIM);
        H1 << offAxisX, offAxisY, 0.0, 0.0, 0.0, 0.0, 0.0;
        
        // 测量2: 装甲板到中心的向量在装甲板法向上的投影等于固定半径276.5 (xy平面)
        double axisX = -sinYaw;     // 装甲板法向单位向量x分量
        double axisY = cosYaw;      // 装甲板法向单位向量y分量
        
        // 测量2的值：装甲板在法向上的投影应为固定半径
        double z2 = axisX * armor_x + axisY * armor_y;
        
        // 测量2的测量矩阵 - 只更新center_x, center_y
        Eigen::RowVectorXd H2(STATE_DIM);
        H2 << axisX, axisY, 0.0, 0.0, 0.0, 0.0, 0.0;
        
        // 测量3: z轴测量 - 装甲板z坐标与中心z坐标的关系
        // armor_z = center_z (因为前哨站vz=0)
        double z3 = armor_z;
        
        // 测量3的测量矩阵 - 只更新center_z
        Eigen::RowVectorXd H3(STATE_DIM);
        H3 << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
        
        // 更新测量1
        double weight1 = weight;
        double S1 = H1 * P_center_ * H1.transpose() + 1.0 / weight1;
        Eigen::VectorXd K1 = P_center_ * H1.transpose() / S1;
        double innovation1 = z1 - H1 * x_center_;
        x_center_ = x_center_ + K1 * innovation1;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        P_center_ = (I - K1 * H1) * P_center_;
        
        // 更新测量2
        double weight2 = weight;
        double S2 = H2 * P_center_ * H2.transpose() + 1.0 / weight2;
        Eigen::VectorXd K2 = P_center_ * H2.transpose() / S2;
        double innovation2 = z2 - H2 * x_center_ + 276.5;  // 固定半径
        x_center_ = x_center_ + K2 * innovation2;
        P_center_ = (I - K2 * H2) * P_center_;
        
        // 更新测量3（z轴测量）
        double weight3 = weight;  // z轴测量权重
        double S3 = H3 * P_center_ * H3.transpose() + 1.0 / weight3;
        Eigen::VectorXd K3 = P_center_ * H3.transpose() / S3;
        double innovation3 = z3 - H3 * x_center_;
        x_center_ = x_center_ + K3 * innovation3;
        P_center_ = (I - K3 * H3) * P_center_;
        
        P_center_ /= std::exp(-lambda_ * dt);

        // 强制固定前哨站模式的状态
        x_center_(3) = 0.0;  // vx = 0
        x_center_(4) = 0.0;  // vy = 0
        x_center_(5) = 0.0;  // vz = 0
        x_center_(6) = 276.5; // r = 276.5
    } else {
        // 测量1的值：装甲板在垂直方向上的投影应为0
        double z1 = offAxisX * armor_x + offAxisY * armor_y;
        
        // 测量1的测量矩阵 (7维)
        Eigen::RowVectorXd H1(STATE_DIM);
        H1 << offAxisX, offAxisY, 0.0, offAxisX * t, offAxisY * t, 0.0, 0.0;
        
        // 测量2: 装甲板到中心的向量在装甲板法向上的投影等于半径r (xy平面)
        // axisVector · armorToCenterVector = r
        double axisX = -sinYaw;     // 装甲板法向单位向量x分量
        double axisY = cosYaw;      // 装甲板法向单位向量y分量
        
        // 测量2的值：装甲板在法向上的投影应为r
        double z2 = axisX * armor_x + axisY * armor_y;
        
        // 测量2的测量矩阵 (7维)
        Eigen::RowVectorXd H2(STATE_DIM);
        H2 << axisX, axisY, 0.0, axisX * t, axisY * t, 0.0, -1.0;
        
        // 测量3: z轴测量 - 装甲板z坐标与中心z坐标的关系
        // armor_z = center_z + center_vz * t (假设装甲板在z方向没有相对运动)
        double z3 = armor_z;
        
        // 测量3的测量矩阵 (7维)
        Eigen::RowVectorXd H3(STATE_DIM);
        H3 << 0.0, 0.0, 1.0, 0.0, 0.0, t, 0.0;
        
        // 正则化测量: 保持r接近上一步的值
        double z4 = r_now_prev_;  // 使用上一步的r值作为正则化目标
        Eigen::RowVectorXd H4 = Eigen::RowVectorXd::Zero(STATE_DIM);
        H4(6) = 1.0;  // 只对r进行正则化
        
        // 更新测量1
        double weight1 = weight;
        double S1 = H1 * P_center_ * H1.transpose() + 1.0 / weight1;
        Eigen::VectorXd K1 = P_center_ * H1.transpose() / S1;
        double innovation1 = z1 - H1 * x_center_;
        x_center_ = x_center_ + K1 * innovation1;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
        P_center_ = (I - K1 * H1) * P_center_;
        
        // 更新测量2
        double weight2 = weight;
        double S2 = H2 * P_center_ * H2.transpose() + 1.0 / weight2;
        Eigen::VectorXd K2 = P_center_ * H2.transpose() / S2;
        double innovation2 = z2 - H2 * x_center_ - delta_r; // 根据两r差异设置目标 算出的x_center_(6)将趋向于原结果+delta_r
        x_center_ = x_center_ + K2 * innovation2;
        P_center_ = (I - K2 * H2) * P_center_;
        
        // 更新测量3（z轴测量）
        double weight3 = weight;  // z轴测量权重
        double S3 = H3 * P_center_ * H3.transpose() + 1.0 / weight3;
        Eigen::VectorXd K3 = P_center_ * H3.transpose() / S3;
        double innovation3 = z3 - H3 * x_center_ - delta_z;
        x_center_ = x_center_ + K3 * innovation3;
        P_center_ = (I - K3 * H3) * P_center_;
        
        // 更新正则化测量
        double weight4 = regularization_weight_;  // 正则化权重
        double S4 = H4 * P_center_ * H4.transpose() + 1.0 / weight4;
        Eigen::VectorXd K4 = P_center_ * H4.transpose() / S4;
        double innovation4 = z4 - H4 * x_center_;
        x_center_ = x_center_ + K4 * innovation4;
        P_center_ = (I - K4 * H4) * P_center_;
        
        P_center_ /= std::exp(-lambda_ * dt);

        // 限制半径在合理范围内
        if (!(x_center_(6) >= 200.0)) x_center_(6) = 200.0; // 限位，同时防止正则化导致nan传播
        if (!(x_center_(6) <= 400.0)) x_center_(6) = 400.0;
    }
}

void RotationMotionModel::updateCenterResult(double current_time) {
    center_x = x_center_(0) + x_center_(3) * current_time;
    center_y = x_center_(1) + x_center_(4) * current_time;
    center_z = x_center_(2) + x_center_(5) * current_time;
    center_vx = x_center_(3);
    center_vy = x_center_(4);
    center_vz = x_center_(5);
    // 更新半径值
    r_now = x_center_(6);
    r_now_prev_ = r_now;  // 保存当前r值用于下一次正则化
}

void RotationMotionModel::update(ObservedData& observedData) {
    if (robust_ekf_active_ && robust_tracker_) {
        const double dt = observedData.t - last_update_time_;

        rm_ekf::ArmorObservation obs;
        obs.x = observedData.x / 1000.0;
        obs.y = observedData.y / 1000.0;
        obs.z = observedData.z / 1000.0;
        obs.yaw = observedData.yaw;

        robust_last_result_ = robust_tracker_->process(obs, dt, -1);
        if (robust_last_result_.updated || robust_last_result_.initialized_this_frame) {
            ++update_frames_count;
        }

        observedData.dt = dt;
        last_observed_data = observedData;
        last_update_time_ = observedData.t;
        syncLegacyFieldsFromRobust();
        return;
    }

    update_frames_count += 1;

    observedData.dt = observedData.t - last_observed_data.t;

    if (angle_ekf_->isYawJumpForRMM(observedData.yaw, observedData.dt)) {
        observedData.yaw_jump = true;
        if (!is_outpost) {
            std::swap(r_now, r_another);
            std::swap(r_now_prev_, r_another_prev_);
            std::swap(center_z, z_another);

            if (debug_flip_flag == 1) {
                debug_flip_flag = 2;
            } else {
                debug_flip_flag = 1;
            }
        }
        std::cout << "RMM Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump! Yaw jump!" << std::endl;
    }

    observedDataHistory.push_back(observedData);
    if (observedDataHistory.size() > max_history) {
        observedDataHistory = std::vector<ObservedData>(
            observedDataHistory.end() - max_history, observedDataHistory.end());
    }
    last_observed_data = observedData;

    int yaw_jump_count = 0;
    for (size_t i = 0; i < observedDataHistory.size(); ++i) {
        const auto& data = observedDataHistory[i];
        if (data.yaw_jump) {
            yaw_jump_count += 1;
        }
    }
    bool this_yaw_jump = yaw_jump_count%2==1;

    // 使用指数衰减最小二乘更新中心状态
    resetExponentialLS();
    double current_time = observedData.t;
    for (size_t i = 0; i < observedDataHistory.size(); ++i) {
        const auto& data = observedDataHistory[i];
        double time_offset = data.t - current_time;  // 相对于当前时间的时间偏移
        // 计算权重：时间越近权重越大
        double time_weight = 1.0;//std::exp(-std::abs(time_offset) * 0.1);
        this_yaw_jump = data.yaw_jump ? (!this_yaw_jump) : this_yaw_jump;
        if (!is_outpost) {
            updateExponentialLS(data.x, data.y, data.z, data.yaw, time_offset, data.dt,
                this_yaw_jump ? time_weight*0.2 : time_weight, 
                this_yaw_jump ? r_now-r_another : 0.0, 
                this_yaw_jump ? center_z-z_another : 0.0);
        } else {
            updateExponentialLS(data.x, data.y, data.z, data.yaw, time_offset, data.dt, time_weight, 0.0, 0.0);
        }
    }
    
    // 获取当前中心状态
    updateCenterResult(0.0);  // 当前时刻
    
    // 使用EKF更新角度和角速度，传入xc, yc, r
    angle_ekf_->update(observedData.yaw, observedData.x, observedData.y, 
                        center_x, center_y, r_now, observedData.dt);
    last_update_time_ = observedData.t;
    rotation_direction = angle_ekf_->getVyaw() >= 0 ? 1.0 : -1.0;
}

void RotationMotionModel::emptyUpdate(double update_time) {
    if (robust_ekf_active_ && robust_tracker_) {
        const double dt = update_time - last_update_time_;
        robust_last_result_ = robust_tracker_->process(std::nullopt, dt, -1);
        last_update_time_ = update_time;
        syncLegacyFieldsFromRobust();
        return;
    }

    PredictResult pred_data_to_update = predict(update_time - last_update_time_);
    ObservedData update_data({
        pred_data_to_update.armors[0].x,
        pred_data_to_update.armors[0].y,
        pred_data_to_update.armors[0].z,
        pred_data_to_update.armors[0].yaw,
        update_time
    });
    update(update_data);
    update_frames_count -= 1;
}

PredictResult RotationMotionModel::predict(double predictTime) {
    PredictResult result;

    if (robust_ekf_active_ && robust_tracker_) {
        const rm_ekf::ArmorState s = robust_tracker_->state();
        result.center_x = (s.x + predictTime * s.vx) * 1000.0;
        result.center_y = (s.y + predictTime * s.vy) * 1000.0;
        result.center_z = (s.z + predictTime * s.vz) * 1000.0;
        result.z_another = (s.z + s.h + predictTime * s.vz) * 1000.0;
        result.r_now = s.r1 * 1000.0;
        result.r_another = s.r2 * 1000.0;
        result.yaw = rm_ekf::wrapAngle(s.yaw + s.w * predictTime);
        result.rotation_direction = (s.w >= 0.0) ? 1 : -1;

        const auto armors = robust_tracker_->predictArmors(predictTime);
        result.armors.reserve(armors.size());
        for (const auto& a : armors) {
            const double r_m = (a.id % 2 == 0) ? s.r1 : s.r2;
            result.armors.push_back(SimpleArmor{
                a.x * 1000.0,
                a.y * 1000.0,
                a.z * 1000.0,
                r_m * 1000.0,
                a.yaw
            });
        }
        return result;
    }
    
    // 使用原始方法预测平移状态
    result.center_x = center_x + predictTime * center_vx;
    result.center_y = center_y + predictTime * center_vy;
    result.center_z = center_z + predictTime * center_vz;
    result.z_another = z_another + predictTime * center_vz;
    result.r_now = r_now;
    result.r_another = r_another;
    
    // 使用EKF预测角度
    if (angle_ekf_->isInitialized()) {
        double ekf_yaw = angle_ekf_->getTotalYaw();
        double ekf_vyaw = angle_ekf_->getVyaw();
        result.yaw = ekf_yaw + ekf_vyaw * predictTime;
        
        // 处理角度环绕
        if (result.yaw > M_PI) result.yaw -= 2.0 * M_PI;
        if (result.yaw < -M_PI) result.yaw += 2.0 * M_PI;
    } else {
        result.yaw = last_observed_data.yaw;
    }
    
    result.rotation_direction = rotation_direction;
    int total_jump_time_with_direction = angle_ekf_->getTotalJumpTimeWithDirection();
    // 生成装甲板预测
    for (int i = 0; i < n_armors; i++) {
        double armor_yaw = result.yaw - i * jump_rad; // double armor_yaw = result.yaw - i * rotation_direction * jump_rad
        double r_using = is_outpost ? r_now : (((i+total_jump_time_with_direction)%2==0) ? r_now : r_another);
        double z_using = is_outpost ? result.center_z : (((i+total_jump_time_with_direction)%2==0) ? result.center_z : result.z_another);
        result.armors.push_back(SimpleArmor({
            result.center_x + r_using * std::sin(armor_yaw),
            result.center_y - r_using * std::cos(armor_yaw),
            z_using,
            r_using,
            armor_yaw
        }));
    }

    return result;
}

RotationMotionState RotationMotionModel::getState() {
    RotationMotionState state{};

    if (robust_ekf_active_ && robust_tracker_) {
        const rm_ekf::ArmorState s = robust_tracker_->state();
        state.center_x = s.x * 1000.0;
        state.center_y = s.y * 1000.0;
        state.center_z = s.z * 1000.0;
        state.z_another = (s.z + s.h) * 1000.0;
        state.center_vx = s.vx * 1000.0;
        state.center_vy = s.vy * 1000.0;
        state.center_vz = s.vz * 1000.0;
        state.r_now = s.r1 * 1000.0;
        state.r_another = s.r2 * 1000.0;
        state.yaw = s.yaw;
        state.total_yaw = s.yaw;
        state.vyaw = s.w;
        state.update_frames = update_frames_count;
        return state;
    }
    state.center_vx = center_vx;
    state.center_vy = center_vy;
    state.center_vz = center_vz;
    state.r_now = r_now;
    state.r_another = r_another;
    state.center_x = center_x;
    state.center_y = center_y;
    state.center_z = center_z;
    state.z_another = z_another;
    state.update_frames = update_frames_count;
    
    if (angle_ekf_->isInitialized()) {
        state.yaw = angle_ekf_->getYaw();
        state.total_yaw = angle_ekf_->getTotalYaw();
        state.vyaw = angle_ekf_->getVyaw();
    } else {
        state.yaw = 0.0;
        state.total_yaw = 0.0;
        state.vyaw = 0.0;
    }
    
    return state;
}

RotationMotionDebugState RotationMotionModel::getDebugState() const {
    RotationMotionDebugState debug;
    debug.robust_ekf_active = robust_ekf_active_;
    if (!robust_ekf_active_) {
        debug.tracker_state = "LEGACY_OUTPOST";
        return debug;
    }

    debug.tracker_state = rm_ekf::trackerStateName(robust_last_result_.state);
    debug.matched_id = robust_last_result_.matched_id;
    debug.measurement_valid = robust_last_result_.measurement_valid;
    debug.updated = robust_last_result_.updated;
    debug.lost_frames = robust_last_result_.lost_frames;
    debug.nis = robust_last_result_.nis;
    debug.position_error_m = robust_last_result_.position_error;
    debug.yaw_error_deg = robust_last_result_.yaw_error >= 0.0
                              ? rm_ekf::rad2deg(robust_last_result_.yaw_error)
                              : -1.0;
    debug.phase_observer_valid = robust_last_result_.phase_observer_valid;
    debug.phase_w_instant = robust_last_result_.phase_w_instant;
    debug.phase_w_filtered = robust_last_result_.phase_w_filtered;
    debug.direction_reversal = robust_last_result_.direction_reversal;
    debug.armor_switched = robust_last_result_.armor_switched;
    debug.recovered = robust_last_result_.recovered;
    debug.phase_w_applied = robust_last_result_.phase_w_applied;
    return debug;
}

bool RotationMotionModel::isTrackerReady() const {
    if (!robust_ekf_active_) return true;
    return robust_tracker_ && robust_tracker_->ready();
}

double RotationMotionModel::getCamToCenterYaw() {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    double cam_to_rotation_center_yaw = std::atan2(-(center_x - cam_center[0]), center_y - cam_center[1]);
    return cam_to_rotation_center_yaw;
}

double RotationMotionModel::getTheoreticYaw(double armor_x, double armor_y) {
    double theoreticYawFacingArmor = getTheoreticYawFacingArmor(armor_x, armor_y);
    double cam_to_rotation_center_yaw = getCamToCenterYaw();
    return cam_to_rotation_center_yaw + theoreticYawFacingArmor;
}

double RotationMotionModel::getTheoreticYawFacingArmor(double armor_x, double armor_y) {
    std::vector<float> cam_center = rest_frame_ -> getCamPosition();
    std::vector<double> cam_to_rotation_center_vector = {center_x - cam_center[0], center_y - cam_center[1]};
    double cam_to_rotation_center_vector_len = std::sqrt(cam_to_rotation_center_vector[0] * cam_to_rotation_center_vector[0] + cam_to_rotation_center_vector[1] * cam_to_rotation_center_vector[1]);
    std::vector<double> right_unit_v = {cam_to_rotation_center_vector[1] / cam_to_rotation_center_vector_len, - cam_to_rotation_center_vector[0] / cam_to_rotation_center_vector_len};
    std::vector<double> center_armor_v = {armor_x - center_x, armor_y - center_y};
    double right_shift = right_unit_v[0] * center_armor_v[0] + right_unit_v[1] * center_armor_v[1];
    if (right_shift > r_now) {
        return M_PI / 2.0;
    }
    if (right_shift < -r_now) {
        return - M_PI / 2.0;
    }
    return std::asin(right_shift / r_now);
}
