#!/usr/bin/env python3
"""
Gazebo在线标定节点

在Gazebo中给Iris发阶跃速度指令，记录实际速度响应。

关键修复：
- CBRK_SUPPLY_CHK=894281（PX4 SITL正确值，不是894415）
- 需要通过pymavlink在GCS端口发heartbeat（PX4要求GCS连接）
- 位置setpoint用于OFFBOARD解锁（速度0可能被忽略）
- ROS2消息字段必须显式转float

使用方法：
1. 终端1: 启动PX4 SITL + Gazebo
2. 终端2: 启动MAVROS
3. 终端3: python src/deploy/calibrate_online.py

输出：
- outputs/logs/calibration_online_*.csv  原始响应数据
"""

import numpy as np
import time
import csv
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import SetMode, CommandBool
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("ERROR: rclpy未安装，无法执行在线标定")
    sys.exit(1)

try:
    from pymavlink import mavutil
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False
    print("WARNING: pymavlink未安装，GCS heartbeat将不可用")

# MAVROS使用BEST_EFFORT QoS，订阅端必须匹配
MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# 安全飞行高度
SAFE_HEIGHT = 5.0


class GCSHeartbeatThread(threading.Thread):
    """
    后台线程：持续发送GCS heartbeat到PX4

    PX4要求GCS连接才能arm（即使NAV_DLL_ACT=0）。
    需要通过PX4的Normal模式端口(14550)发送MAV_TYPE_GCS heartbeat。
    """

    def __init__(self, port=14550):
        super().__init__(daemon=True)
        self.port = port
        self.running = False
        self.connected = False

    def run(self):
        if not HAS_PYMAVLINK:
            print("GCS heartbeat: pymavlink不可用，跳过")
            return

        self.running = True
        try:
            # 绑定GCS端口，PX4会发heartbeat到14550
            self.mav = mavutil.mavlink_connection(
                f'udp:127.0.0.1:{self.port}',
                source_system=255,
                source_component=0
            )
            # 等待PX4的heartbeat
            hb = self.mav.wait_heartbeat(timeout=10)
            if hb:
                self.connected = True
                print(f"GCS heartbeat: 已连接到PX4 (sys={self.mav.target_system})")

            while self.running:
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                time.sleep(1.0)
        except Exception as e:
            print(f"GCS heartbeat错误: {e}")

    def stop(self):
        self.running = False


class CalibrationNode(Node):
    """PX4阶跃响应标定节点"""

    def __init__(self):
        super().__init__('calibration_node')

        # 状态
        self.current_state = None
        self.current_pose = None
        self.current_vel = None
        self.armed = False
        self.offboard = False
        self.current_position = np.array([0.0, 0.0, 0.0])

        # 订阅（MAVROS用BEST_EFFORT，必须匹配QoS）
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb, MAVROS_QOS
        )
        self.vel_sub = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', self.vel_cb, MAVROS_QOS
        )

        # 发布位置setpoint（用于解锁和起飞）
        self.pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10
        )
        # 发布速度指令（用于阶跃响应测试）
        self.vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10
        )

        # 服务客户端
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        # 标定数据
        self.data = []
        self.recording = False

        # GCS heartbeat线程
        self.gcs_heartbeat = GCSHeartbeatThread()

        self.get_logger().info('标定节点已初始化')

    def state_cb(self, msg):
        self.current_state = msg
        self.armed = msg.armed
        self.offboard = (msg.mode == 'OFFBOARD')

    def pose_cb(self, msg):
        self.current_pose = msg
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def vel_cb(self, msg):
        self.current_vel = msg

    def wait_for_connection(self, timeout=30.0):
        """等待MAVROS连接"""
        start = time.time()
        while not self.current_state or not self.current_state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return False
        self.get_logger().info('MAVROS已连接')
        return True

    def set_offboard_mode(self):
        """切换OFFBOARD模式"""
        req = SetMode.Request()
        req.custom_mode = 'OFFBOARD'
        self.set_mode_cli.call_async(req)

    def arm(self):
        """解锁"""
        req = CommandBool.Request()
        req.value = True
        self.arm_cli.call_async(req)

    def disarm(self):
        """上锁"""
        req = CommandBool.Request()
        req.value = False
        self.arm_cli.call_async(req)

    def send_position_setpoint(self, x, y, z):
        """发送位置setpoint"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.pos_pub.publish(msg)

    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """发送速度指令（显式转float避免ROS2类型断言失败）"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(msg)

    def get_current_velocity(self):
        """获取当前速度"""
        if self.current_vel is not None:
            return np.array([
                self.current_vel.twist.linear.x,
                self.current_vel.twist.linear.y,
                self.current_vel.twist.linear.z,
            ])
        return np.zeros(3)

    def set_px4_params(self):
        """设置PX4参数以允许OFFBOARD解锁"""
        from mavros_msgs.srv import ParamSetV2
        from rcl_interfaces.msg import ParameterValue as PV

        param_cli = self.create_client(ParamSetV2, '/mavros/param/set')
        param_cli.wait_for_service(timeout_sec=10.0)

        # 关键参数说明：
        # CBRK_SUPPLY_CHK=894281 是PX4 SITL的magic number（不是894415！）
        # NAV_DLL_ACT=0 禁用数据链路丢失动作（不要求GCS连接）
        # COM_RC_IN_MODE=1 是PX4 SITL默认值（允许无RC）
        params = {
            'COM_ARM_WO_GPS': 1,       # 允许无GPS解锁
            'COM_RC_IN_MODE': 1,       # 无需RC（PX4 SITL默认值）
            'CBRK_SUPPLY_CHK': 894281, # 禁用电源检查（SITL正确值！）
            'COM_ARM_AUTH_REQ': 0,     # 无需授权
            'COM_LOW_BAT_ACT': 0,      # 低电量无动作
            'NAV_RCL_ACT': 0,          # RC丢失不要求
            'NAV_DLL_ACT': 0,          # 数据链路丢失不要求
        }

        for name, value in params.items():
            req = ParamSetV2.Request()
            req.param_id = name
            val = PV()
            val.type = 2  # INTEGER
            val.integer_value = value
            req.value = val
            future = param_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            result = future.result()
            status = 'OK' if result.success else 'FAIL'
            self.get_logger().info(f'  {name}={value} -> {status}')

        self.destroy_client(param_cli)

    def offboard_arm_and_takeoff(self, target_height=SAFE_HEIGHT, timeout=30.0):
        """
        用位置setpoint执行OFFBOARD解锁+起飞

        关键步骤：
        1. 启动GCS heartbeat线程（PX4要求GCS连接）
        2. 设置PX4参数
        3. 持续发送位置setpoint（>2Hz）
        4. 切换OFFBOARD模式
        5. 发送arm命令
        """
        # 启动GCS heartbeat
        self.gcs_heartbeat.start()
        time.sleep(3.0)  # 等GCS连接建立

        self.get_logger().info(f'OFFBOARD解锁+起飞到{target_height}m...')

        # 1. 用位置setpoint预热（PX4要求OFFBOARD前就有setpoint流）
        self.get_logger().info('预热位置setpoint (5秒)...')
        start = time.time()
        while time.time() - start < 5.0:
            self.send_position_setpoint(0, 0, target_height)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)  # 20Hz

        # 2. 同时发送OFFBOARD+ARM+setpoint
        self.get_logger().info('切换OFFBOARD并解锁...')
        last_arm_time = 0
        last_mode_time = 0
        start = time.time()
        while time.time() - start < timeout:
            now = time.time()
            # 持续发位置setpoint（必须保持>2Hz）
            self.send_position_setpoint(0, 0, target_height)

            # 每0.5秒请求一次解锁和模式切换
            if now - last_arm_time > 0.5:
                if not self.armed:
                    self.arm()
                last_arm_time = now
            if now - last_mode_time > 0.5:
                if not self.offboard:
                    self.set_offboard_mode()
                last_mode_time = now

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

            if self.armed and self.offboard:
                break

        if not self.armed or not self.offboard:
            self.get_logger().error(
                f'无法OFFBOARD解锁 (armed={self.armed}, offboard={self.offboard})')
            return False

        self.get_logger().info('已解锁并切换OFFBOARD，等待起飞...')

        # 3. 等待到达目标高度
        start = time.time()
        while time.time() - start < 15.0:
            self.send_position_setpoint(0, 0, target_height)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

            if self.current_position[2] > target_height - 0.5:
                self.get_logger().info(
                    f'到达目标高度: {self.current_position[2]:.2f}m')
                return True

        self.get_logger().warn(
            f'起飞超时，当前高度: {self.current_position[2]:.2f}m')
        return True  # 仍然继续

    def run_step_response(self, axis='x', vel_cmd=3.0, duration=10.0, dt=0.05):
        """
        执行阶跃响应测试

        Args:
            axis: 测试轴 'x', 'y', 'z'
            vel_cmd: 阶跃速度 m/s
            duration: 持续时间 s
            dt: 采样间隔 s
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis]

        self.get_logger().info(f'开始阶跃响应测试: {axis}轴, {vel_cmd} m/s, {duration}s')

        # 先悬停2秒
        self.get_logger().info('悬停2秒...')
        start = time.time()
        while time.time() - start < 2.0:
            self.send_velocity(0, 0, 0)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        # 发阶跃指令
        step_data = []
        self.recording = True
        start = time.time()

        while time.time() - start < duration:
            # 发送速度指令
            vel_msg = [0.0, 0.0, 0.0]
            vel_msg[axis_idx] = vel_cmd
            self.send_velocity(*vel_msg)

            # 记录数据
            rclpy.spin_once(self, timeout_sec=0.01)
            current_vel = self.get_current_velocity()
            elapsed = time.time() - start
            step_data.append({
                'time': elapsed,
                'vx': current_vel[0],
                'vy': current_vel[1],
                'vz': current_vel[2],
                'cmd_axis': axis,
                'cmd_vel': vel_cmd,
            })

            time.sleep(dt)

        self.recording = False
        self.get_logger().info(f'阶跃响应测试完成, {len(step_data)}个数据点')

        # 停止
        self.send_velocity(0, 0, 0)

        # 累积数据（不清空之前的数据）
        self.data.extend(step_data)

    def save_data(self, filename=None):
        """保存标定数据到CSV"""
        if not self.data:
            self.get_logger().warn('无数据可保存')
            return

        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'outputs', 'logs'
        )
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            filename = f'calibration_online_{time.strftime("%Y%m%d_%H%M%S")}.csv'

        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'vx', 'vy', 'vz', 'cmd_axis', 'cmd_vel'])
            writer.writeheader()
            writer.writerows(self.data)

        self.get_logger().info(f'标定数据已保存: {filepath}')
        return filepath


def main():
    rclpy.init()
    node = CalibrationNode()

    # 1. 等待连接
    if not node.wait_for_connection():
        node.get_logger().error('MAVROS连接超时')
        return

    # 2. 等待服务可用
    node.get_logger().info('等待MAVROS服务可用...')
    node.set_mode_cli.wait_for_service(timeout_sec=10.0)
    node.arm_cli.wait_for_service(timeout_sec=10.0)

    # 3. 设置PX4参数
    node.get_logger().info('设置PX4参数...')
    node.set_px4_params()

    # 4. OFFBOARD解锁+起飞
    if not node.offboard_arm_and_takeoff(target_height=SAFE_HEIGHT):
        node.get_logger().error('起飞失败，退出')
        node.gcs_heartbeat.stop()
        return

    # 5. 执行各轴阶跃响应
    test_configs = [
        ('x', 3.0),   # x轴 3m/s
        ('x', -2.0),  # x轴 -2m/s
        ('y', 2.0),   # y轴 2m/s
        ('z', 1.0),   # z轴 1m/s
    ]

    for axis, vel in test_configs:
        node.run_step_response(axis=axis, vel_cmd=vel, duration=8.0)
        # 间隔2秒悬停
        start = time.time()
        while time.time() - start < 2.0:
            node.send_velocity(0, 0, 0)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.05)

    # 6. 保存数据
    filepath = node.save_data()

    # 7. 降落
    node.get_logger().info('降落...')
    start = time.time()
    while time.time() - start < 15.0:
        node.send_position_setpoint(0, 0, 0.5)
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(0.05)
        if node.current_position[2] < 0.3:
            break

    # 上锁
    node.disarm()

    # 停止GCS heartbeat
    node.gcs_heartbeat.stop()

    # 8. 输出后续步骤
    if filepath:
        node.get_logger().info(f'标定数据已保存，请运行离线拟合:')
        node.get_logger().info(f'  python src/deploy/calibrate_dynamics.py --data {filepath}')

    rclpy.shutdown()


if __name__ == "__main__":
    main()
