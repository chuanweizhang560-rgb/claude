import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.controllers.base_controller import RuleController


class TurbineOrbitController(RuleController):
    """
    风机螺旋环绕控制器

    primary角色：螺旋上升，半径8m，yaw指向塔心
    assistant角色：在轮毂高度悬停，小半径(5m)缓慢旋转

    控制策略：PD追踪轨道点，轨道点沿圆柱面缓慢移动
    """

    def __init__(self, config: dict = None):
        self.orbit_radius = 8.0         # primary环绕半径
        self.assistant_radius = 5.0     # assistant环绕半径
        self.start_height = 3.0         # primary起始环绕高度（从底部开始）
        self.hub_height_offset = 5.0    # assistant在轮毂上方偏移
        self.angular_speed = 0.12       # rad/s 环绕角速度
        self.climb_speed = 0.8          # m/s 上升速度（70m/0.8=87.5s完成一次螺旋）
        self.approach_speed = 5.0       # 接近速度 m/s
        self.dt = 0.05

        # PD追踪增益
        self.kp_track = 2.0             # 位置追踪P增益
        self.kd_track = 0.3             # 位置追踪D增益

        # 内部状态
        self.orbit_angle = 0.0          # 当前环绕角度
        self.orbit_height = self.start_height  # 当前环绕高度

        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def reset(self):
        """重置环绕状态"""
        self.orbit_angle = 0.0
        self.orbit_height = self.start_height

    def get_guided_velocity(self, state: dict) -> np.ndarray:
        pos = np.asarray(state['pos'], dtype=np.float64)
        vel = np.asarray(state['vel'], dtype=np.float64)
        yaw = state.get('yaw', 0.0)
        target = state['target']
        role = state.get('role', 'primary')

        turbine_pos = np.asarray(target['position'], dtype=np.float64)
        turbine_height = target.get('height', 70.0)
        hub_pos = turbine_pos.copy()
        hub_pos[2] += turbine_height

        # 1. 计算到风机的XY距离
        to_turbine = turbine_pos - pos
        dist_xy = np.linalg.norm(self._vec2d(to_turbine))

        # 2. 接近阶段：距离太远时先飞向轨道起始点
        approach_dist = self.orbit_radius + 5.0
        if dist_xy > approach_dist:
            # 飞向轨道起始点（风机底部附近的轨道位置）
            approach_target = np.array([
                turbine_pos[0] + self.orbit_radius,  # 轨道起始点（angle=0时在x正方向）
                turbine_pos[1],
                turbine_pos[2] + self.start_height,
            ])
            to_approach = approach_target - pos
            dist_approach = np.linalg.norm(to_approach)
            if dist_approach > 1.0:
                # PD追踪
                vel_cmd = self.kp_track * to_approach / dist_approach * min(dist_approach, self.approach_speed) - self.kd_track * vel[:3]
            else:
                vel_cmd = np.zeros(3)

            target_yaw = self._yaw_to_target(pos, turbine_pos)
            yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)
            return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])

        # 3. 环绕阶段：PD追踪移动的轨道点
        self.orbit_angle += self.angular_speed * self.dt

        if role == 'primary':
            # 计算轨道目标点
            orbit_target = np.array([
                turbine_pos[0] + self.orbit_radius * np.cos(self.orbit_angle),
                turbine_pos[1] + self.orbit_radius * np.sin(self.orbit_angle),
                turbine_pos[2] + self.orbit_height,
            ])

            # 更新轨道高度（螺旋上升）
            self.orbit_height += self.climb_speed * self.dt
            if self.orbit_height > turbine_height:
                self.orbit_height = turbine_height

            # PD追踪轨道目标点
            error = orbit_target - pos
            dist_to_target = np.linalg.norm(error)

            if dist_to_target > 0.1:
                # 期望速度 = 轨道点移动速度 + PD修正
                # 轨道点速度（切线方向 + 上升）
                orbit_vel = np.array([
                    -self.orbit_radius * self.angular_speed * np.sin(self.orbit_angle),
                    self.orbit_radius * self.angular_speed * np.cos(self.orbit_angle),
                    self.climb_speed,
                ])
                # 前馈 + PD修正
                vel_cmd = orbit_vel + self.kp_track * error - self.kd_track * vel[:3]
            else:
                # 已在轨道点上，用轨道速度
                vel_cmd = np.array([
                    -self.orbit_radius * self.angular_speed * np.sin(self.orbit_angle),
                    self.orbit_radius * self.angular_speed * np.cos(self.orbit_angle),
                    self.climb_speed,
                ])

        else:
            # assistant：在轮毂高度悬停绕小圈
            orbit_target = np.array([
                turbine_pos[0] + self.assistant_radius * np.cos(self.orbit_angle * 0.5),
                turbine_pos[1] + self.assistant_radius * np.sin(self.orbit_angle * 0.5),
                hub_pos[2] + self.hub_height_offset,
            ])

            orbit_vel = np.array([
                -self.assistant_radius * self.angular_speed * 0.5 * np.sin(self.orbit_angle * 0.5),
                self.assistant_radius * self.angular_speed * 0.5 * np.cos(self.orbit_angle * 0.5),
                0.0,
            ])

            error = orbit_target - pos
            vel_cmd = orbit_vel + self.kp_track * error - self.kd_track * vel[:3]

        # 4. yaw始终指向塔心
        target_yaw = self._yaw_to_target(pos, turbine_pos)
        yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)

        return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])


if __name__ == "__main__":
    print("=== TurbineOrbitController 测试 ===\n")

    turbine_pos = np.array([50.0, 0.0, 0.0])
    target = {'type': 'turbine', 'position': turbine_pos, 'radius': 1.5, 'height': 70.0}
    ctrl = TurbineOrbitController()

    # 测试1: primary远处接近
    state_far = {
        'pos': np.array([50.0, 30.0, 30.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'target': target,
        'role': 'primary',
    }
    vel_far = ctrl.get_guided_velocity(state_far)
    print(f"primary远处接近: vel={vel_far}, 应朝风机飞")

    # 测试2: primary环绕中
    ctrl2 = TurbineOrbitController()
    ctrl2.orbit_angle = 0.0
    ctrl2.orbit_height = 30.0
    state_orbit = {
        'pos': np.array([58.0, 0.0, 30.0]),  # 在环绕半径附近
        'vel': np.zeros(3),
        'yaw': 0.0,
        'target': target,
        'role': 'primary',
    }
    vel_orbit = ctrl2.get_guided_velocity(state_orbit)
    print(f"primary环绕中: vel={vel_orbit}, 应有切线速度+上升+追踪修正")

    # 测试3: assistant
    ctrl3 = TurbineOrbitController()
    ctrl3.orbit_angle = 0.0
    state_assist = {
        'pos': np.array([55.0, 0.0, 75.0]),  # 在轮毂附近
        'vel': np.zeros(3),
        'yaw': 0.0,
        'target': target,
        'role': 'assistant',
    }
    vel_assist = ctrl3.get_guided_velocity(state_assist)
    print(f"assistant: vel={vel_assist}, 应缓慢旋转+vz≈0")

    print("\n=== 测试完成 ===")
