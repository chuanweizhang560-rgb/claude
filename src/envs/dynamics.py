import numpy as np


class QuadrotorDynamics:
    """简化四旋翼动力学：一阶速度响应+阻力"""

    def __init__(self, config: dict = None):
        # 默认参数（从PX4 Gazebo SITL阶跃响应标定拟合）
        # 水平轴: Kp=0.645, tau=1.5s, 稳态≈98.5%指令
        # 垂直轴: Kp=1.57, tau=0.63s, 稳态≈99.4%指令
        self.mass = 1.5          # kg
        self.kp = 0.6452         # 水平速度响应增益（Gazebo标定）
        self.kd = 0.01           # 阻力系数（Gazebo标定，极小）
        self.kp_z = 1.57         # 垂直速度响应增益（Gazebo标定）
        self.kd_z = 0.01         # 垂直阻力系数（Gazebo标定）
        self.max_vel = 5.0       # 最大水平速度 m/s
        self.max_vel_z = 3.0     # 最大垂直速度 m/s
        self.max_yaw_rate = 1.0  # 最大偏航角速率 rad/s
        self.dt = 0.05           # 仿真步长 20Hz

        # 风力影响系数
        self.k_wind = 0.3        # 风力对加速度的影响系数

        # 功耗模型参数
        self.P_hover = 0.0167    # 悬停功耗 %/s（约1%/min，对应20min续航）
        self.k_speed = 0.003     # 速度功耗系数 %/s per (m/s)
        self.k_climb = 0.005     # 爬升功耗系数 %/s per (m/s)
        self.k_wind_power = 0.002  # 抗风功耗系数 %/s per (m/s)

        # 用配置字典覆盖默认参数
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def step(self, state: dict, vel_cmd: np.ndarray, wind: np.ndarray = None) -> dict:
        """
        执行一步动力学更新

        Args:
            state: {'pos': [x,y,z], 'vel': [vx,vy,vz], 'yaw': float}
            vel_cmd: [vx, vy, vz, yaw_rate] 目标速度指令
            wind: [wx, wy, wz] 当前风速（可选）

        Returns:
            新的 state dict
        """
        pos = np.array(state['pos'], dtype=np.float64)
        vel = np.array(state['vel'], dtype=np.float64)
        yaw = float(state['yaw'])

        vel_cmd = np.array(vel_cmd, dtype=np.float64)

        # 1. 水平轴响应：加速度 = kp*(目标速度-当前速度) - kd*当前速度
        acc_h = self.kp * (vel_cmd[:2] - vel[:2]) - self.kd * vel[:2]

        # 2. 垂直轴响应：使用kp_z和kd_z
        acc_z = self.kp_z * (vel_cmd[2] - vel[2]) - self.kd_z * vel[2]

        acc = np.array([acc_h[0], acc_h[1], acc_z])

        # 2. 风力影响：加速度额外项 = k_wind * (wind - velocity)
        if wind is not None:
            wind = np.array(wind, dtype=np.float64)
            acc[:2] += self.k_wind * (wind[:2] - vel[:2])
            acc[2] += self.k_wind * (wind[2] - vel[2])

        # 3. 速度积分
        vel = vel + acc * self.dt

        # 4. 速度限幅
        # 水平限幅：norm([vx,vy]) <= max_vel
        h_speed = np.linalg.norm(vel[:2])
        if h_speed > self.max_vel:
            vel[:2] = vel[:2] * (self.max_vel / h_speed)

        # 垂直限幅：|vz| <= max_vel_z
        vel[2] = np.clip(vel[2], -self.max_vel_z, self.max_vel_z)

        # 5. 位置积分
        pos = pos + vel * self.dt

        # 6. 偏航角积分（限幅偏航角速率）
        yaw_rate = np.clip(vel_cmd[3], -self.max_yaw_rate, self.max_yaw_rate)
        yaw = yaw + yaw_rate * self.dt

        # 偏航角归一化到 [-pi, pi]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        return {
            'pos': pos.tolist(),
            'vel': vel.tolist(),
            'yaw': yaw
        }

    def get_power(self, velocity: np.ndarray, wind: np.ndarray = None) -> float:
        """
        计算当前功耗（用于电量模型）

        Args:
            velocity: [vx,vy,vz] 当前速度
            wind: [wx,wy,wz] 当前风速（可选）

        Returns:
            power: 功耗（百分比/秒），悬停约1%/min → 0.0167%/s
        """
        velocity = np.array(velocity, dtype=np.float64)

        # 基础悬停功耗
        power = self.P_hover

        # 速度相关功耗
        power += self.k_speed * np.linalg.norm(velocity)

        # 爬升功耗（仅向上爬升时额外消耗）
        power += self.k_climb * max(velocity[2], 0.0)

        # 抗风功耗
        if wind is not None:
            wind = np.array(wind, dtype=np.float64)
            power += self.k_wind_power * np.linalg.norm(wind - velocity)

        return power


if __name__ == "__main__":
    dynamics = QuadrotorDynamics()

    # ========== 测试1：阶跃响应测试 ==========
    print("=" * 60)
    print("测试1：阶跃响应 - 速度指令 [2, 0, 0, 0]")
    print("=" * 60)

    state = {'pos': [0.0, 0.0, 0.0], 'vel': [0.0, 0.0, 0.0], 'yaw': 0.0}
    vel_cmd = np.array([2.0, 0.0, 0.0, 0.0])

    print(f"{'步数':>4}  {'vx':>8}  {'vy':>8}  {'vz':>8}  {'x':>8}  {'y':>8}  {'z':>8}")
    print("-" * 60)

    for i in range(50):
        state = dynamics.step(state, vel_cmd)
        if i % 5 == 0 or i == 49:
            print(f"{i+1:4d}  {state['vel'][0]:8.4f}  {state['vel'][1]:8.4f}  "
                  f"{state['vel'][2]:8.4f}  {state['pos'][0]:8.4f}  "
                  f"{state['pos'][1]:8.4f}  {state['pos'][2]:8.4f}")

    # 检查稳态速度：一阶系统稳态 vel = kp/(kp+kd) * cmd
    expected_steady = dynamics.kp / (dynamics.kp + dynamics.kd) * 2.0
    actual_vx = state['vel'][0]
    print(f"\n稳态速度分析：")
    print(f"  理论稳态速度 = kp/(kp+kd) * cmd = {dynamics.kp}/{dynamics.kp+dynamics.kd} * 2.0 = {expected_steady:.4f} m/s")
    print(f"  实际稳态速度 = {actual_vx:.4f} m/s")
    print(f"  误差 = {abs(actual_vx - expected_steady):.6f} m/s")

    # ========== 测试2：悬停功耗测试 ==========
    print("\n" + "=" * 60)
    print("测试2：悬停功耗 - 静止状态")
    print("=" * 60)

    hover_power = dynamics.get_power(np.array([0.0, 0.0, 0.0]))
    print(f"  悬停功耗 = {hover_power:.4f} %/s")
    print(f"  预期值   = {dynamics.P_hover:.4f} %/s")
    print(f"  换算     = {hover_power * 60:.4f} %/min")
    print(f"  预计续航 = {100.0 / (hover_power * 60):.1f} min")

    # ========== 测试3：全速功耗测试 ==========
    print("\n" + "=" * 60)
    print("测试3：全速水平飞行功耗 - 5 m/s")
    print("=" * 60)

    full_speed_power = dynamics.get_power(np.array([5.0, 0.0, 0.0]))
    print(f"  全速功耗 = {full_speed_power:.4f} %/s")
    print(f"  换算     = {full_speed_power * 60:.4f} %/min")
    print(f"  预计续航 = {100.0 / (full_speed_power * 60):.1f} min")
    print(f"  功耗明细：")
    print(f"    悬停功耗 = {dynamics.P_hover:.4f} %/s")
    print(f"    速度功耗 = {dynamics.k_speed * 5.0:.4f} %/s")
    print(f"    合计     = {dynamics.P_hover + dynamics.k_speed * 5.0:.4f} %/s")
