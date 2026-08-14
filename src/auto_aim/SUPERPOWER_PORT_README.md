# SuperPower EKF 移植说明

当前 `EKFTargetPredictor` 的运行时估计后端已经切换为：

```text
EKFTargetPredictor (单位 / 时间 / yaw 约定适配)
        ↓
sp_ekf::Tracker
        ↓
sp_ekf::Target
        ↓
sp_ekf::ExtendedKalmanFilter
```

旧 `RobustArmorTracker.cpp` 仍保留用于回滚和历史调试类型，但已从 `CMakeLists.txt`
的生产目标移除，`EKFTargetPredictor` 不再读取 `robust_ekf` 参数。

## 对齐的 SuperPower 行为

代码基线固定为 `TongjiSuperPower/sp_vision_25` commit
`ce3e1ce05ea57bef9813bba0c450ff158388f0a2` 的普通四装甲车辆路径：

- 11D 状态 `[x,vx,y,vy,z,vz,a,w,r,l,h]`
- `l = r2 - r1`
- `F/Q` Piecewise White Noise，普通车辆 `v1=100, v2=400`
- `z=[bearing_yaw,pitch,distance,armor_angle]`
- SuperPower 原动态 `R`
- 最近 3 个预测装甲板 + `armor angle error + bearing error` 数据关联
- Joseph 形式协方差更新
- 原 NIS/NEES 统计与最近 100 次 NIS failure 健康检查
- `LOST → DETECTING → TRACKING → TEMP_LOST`
- `min_detect_count=5`, `max_temp_lost_count=15`
- `dt > 0.1 s` 强制重新捕获
- 普通四装甲初始化半径 `0.2 m`

## 必要适配

项目原 yaw 定义和 SuperPower 相差 `pi/2`。只在 `EKFTargetPredictor` 边界转换：

```text
SP_angle = project_yaw + pi/2
project_yaw = SP_angle - pi/2
```

这样项目原几何

```text
armor = center + r * [sin(yaw), -cos(yaw)]
```

与 SuperPower 几何

```text
armor = center - r * [cos(angle), sin(angle)]
```

严格等价。SP estimator 内部没有任何坐标约定补丁。

当前上层仍只向 EKF 提供一块选中的 PnP 装甲板，因此没有复制 SuperPower 的敌方颜色、
多兵种优先级、全向感知 switching，以及 Base/Outpost/Balance 特判。完整差异见
`SUPERPOWER_NOTICE.md`。

## 验证

本次移植已验证：

- SP 核心三层使用 `-Wall -Wextra -Wpedantic -Werror` 编译通过
- 5 帧确认进入 TRACKING
- E0 → E1 的 90° 物理装甲板切换匹配为 `id=1`
- TEMP_LOST → TRACKING 恢复
- `dt > 0.1 s` 回到 DETECTING 重新捕获
- 1000 组随机 yaw/radius 的项目 ↔ SP 几何转换数值一致

Ubuntu/Eigen 环境可单独运行：

```bash
cd src/auto_aim
bash tests/run_superpower_ekf_smoke.sh
```

完整 ROS2 工程仍应在实际工作区执行：

```bash
colcon build --packages-select auto_aim --symlink-install
```
