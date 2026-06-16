# 阶段0.3 PX4动力学标定日志

**日期**: 2026-06-16
**状态**: ✅ 完成

## 目标
通过PX4 SITL + Gazebo Classic在线标定，获取简化动力学模型的Kp/Kd参数

## 标定方法
1. 启动PX4 SITL + Gazebo Classic + MAVROS
2. OFFBOARD模式解锁起飞到5m
3. 分别对各轴发阶跃速度指令（x: ±3/2 m/s, y: 2 m/s, z: 1 m/s）
4. 记录实际速度响应，最小二乘拟合 Kp/Kd

## 踩坑记录（6个关键问题）

### 问题1: ROS2消息类型断言失败
- **现象**: `AssertionError: The 'x' field must be of type 'float'`
- **原因**: Python int/np类型传给ROS2 Vector3字段
- **修复**: `msg.twist.linear.x = float(vx)`

### 问题2: MAVROS QoS不兼容
- **现象**: 订阅pose/vel topic无数据
- **原因**: MAVROS用BEST_EFFORT，默认订阅用RELIABLE
- **修复**: 订阅时使用`QoSProfile(reliability=BEST_EFFORT)`

### 问题3: OFFBOARD模式无法arm (result=1 DENIED)
- **现象**: 可以切OFFBOARD但arm被拒绝
- **根因1**: `CBRK_SUPPLY_CHK=894415`是错误值，PX4 SITL需用`894281`
- **根因2**: PX4要求GCS连接，需要pymavlink在端口14550发MAV_TYPE_GCS heartbeat
- **根因3**: `NAV_DLL_ACT`需设为0禁用数据链路检查
- **修复**: 设置正确参数 + 启动GCS heartbeat后台线程

### 问题4: conda环境Python版本不兼容ROS2
- **现象**: conda Python 3.13无法import rclpy
- **修复**: 重建conda环境Python 3.10 + .pth文件链接ROS2包

### 问题5: MAVROS ParamSetV2类型错误
- **现象**: `ParamValue`对象不能赋给`ParamSetV2.Request.value`
- **原因**: 需要用`rcl_interfaces.msg.ParameterValue`而非`mavros_msgs.msg.ParamValue`
- **修复**: 使用正确的ParameterValue类型

### 问题6: 阶跃响应数据被覆盖
- **现象**: save_data只保存最后一个轴的数据
- **原因**: `run_step_response`每次重置`self.data = []`
- **修复**: 改用`step_data`局部变量，最后`self.data.extend(step_data)`

## 标定结果

| 测试 | Kp | Kd | tau(s) | 稳态/指令 | RMSE(m/s) |
|------|------|------|--------|-----------|-----------|
| x+3m/s | 0.652 | 0.01 | 1.51 | 98.5% | 0.277 |
| x-2m/s | 0.644 | 0.01 | 1.53 | 98.5% | 0.182 |
| y+2m/s | 0.640 | 0.01 | 1.54 | 98.5% | 0.194 |
| z+1m/s | 1.570 | 0.01 | 0.63 | 99.4% | 0.124 |

**结论**:
- 水平轴: Kp≈0.645, Kd≈0.01, 时间常数≈1.5s
- 垂直轴: Kp≈1.57, Kd≈0.01, 时间常数≈0.63s
- PX4速度控制器接近纯一阶响应，阻尼极小
- 之前模拟参数Kp=2.59, Kd=0.84完全偏离实际

## 更新文件
- `src/envs/dynamics.py`: kp=0.6452, kd=0.01, kp_z=1.57, kd_z=0.01, 支持水平/垂直不同参数
- `src/deploy/calibrate_online.py`: 完全重写，集成GCS heartbeat + 正确参数 + QoS修复

## 输出文件
- `outputs/logs/calibration_online_20260616_175109.csv`: 636个数据点
- `outputs/figures/calibration_gazebo_result.png`: 阶跃响应对比图

## 待办
- 电池功耗模型校准（悬停99.8min偏大，应约20min）
