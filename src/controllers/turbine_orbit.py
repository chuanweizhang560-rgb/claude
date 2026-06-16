import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.controllers.base_controller import RuleController


class TurbineOrbitController(RuleController):
    """
    风机螺旋环绕控制器

    primary角色：螺旋上升，半径10m，yaw指向塔心
    assistant角色：在轮毂高度悬停，小半径(5m)缓慢旋转
    """

    def __init__(self, config: dict = None):
        self.orbit_radius = 10.0        # primary环绕半径
        self.assistant_radius = 5.0     # assistant环绕半径
        self.start_height = 5.0         # primary起始环绕高度
        self.hub_height_offset = 5.0    # assistant在轮毂上方偏移
        self.angular_speed = 0.1        # rad/s 环绕角速度
        self.climb_speed = 0.5          # m/s 上升速度
        self.approach_speed = 3.0       # 接近速度 m/s
        self.dt = 0.05

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

        # 1. 计算到风机的距离
        to_turbine = turbine_pos - pos
        dist_xy = np.linalg.norm(self._vec2d(to_turbine))
        dist_3d = np.linalg.norm(to_turbine)

        # 2. 接近阶段：距离太远时先飞向风机
        approach_dist = self.orbit_radius + 5.0
        if dist_xy > approach_dist:
            # 飞向风机上方起始高度
            approach_target = turbine_pos.copy()
            approach_target[2] = self.start_height + turbine_pos[2]
            to_approach = approach_target - pos
            dist_approach = np.linalg.norm(to_approach)
            if dist_approach > 1.0:
                speed = min(self.approach_speed, dist_approach)
                vel_cmd = to_approach / dist_approach * speed
            else:
                vel_cmd = np.zeros(3)

            target_yaw = self._yaw_to_target(pos, turbine_pos)
            yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)
            return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])

        # 3. 环绕阶段
        self.orbit_angle += self.angular_speed * self.dt

        if role == 'primary':
            # primary：螺旋上升
            # 目标点在圆柱面上
            target_on_orbit = np.array([
                turbine_pos[0] + self.orbit_radius * np.cos(self.orbit_angle),
                turbine_pos[1] + self.orbit_radius * np.sin(self.orbit_angle),
                turbine_pos[2] + self.orbit_height,
            ])
            self.orbit_height += self.climb_speed * self.dt
            # 到顶后停止上升
            if self.orbit_height > turbine_height:
                self.orbit_height = turbine_height

            # 速度：切线方向 + 上升
            tangent = np.array([
                -self.orbit_radius * self.angular_speed * np.sin(self.orbit_angle),
                self.orbit_radius * self.angular_speed * np.cos(self.orbit_angle),
                self.climb_speed,
            ])
            vel_cmd = tangent

        else:
            # assistant：在轮毂高度悬停绕小圈
            target_on_orbit = np.array([
                turbine_pos[0] + self.assistant_radius * np.cos(self.orbit_angle * 0.5),
                turbine_pos[1] + self.assistant_radius * np.sin(self.orbit_angle * 0.5),
                hub_pos[2] + self.hub_height_offset,
            ])
            tangent = np.array([
                -self.assistant_radius * self.angular_speed * 0.5 * np.sin(self.orbit_angle * 0.5),
                self.assistant_radius * self.angular_speed * 0.5 * np.cos(self.orbit_angle * 0.5),
                0.0,
            ])
            vel_cmd = tangent

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
        'pos': np.array([60.0, 0.0, 30.0]),  # 在环绕半径上
        'vel': np.zeros(3),
        'yaw': 0.0,
        'target': target,
        'role': 'primary',
    }
    vel_orbit = ctrl2.get_guided_velocity(state_orbit)
    print(f"primary环绕中: vel={vel_orbit}, 应有切线速度+上升")

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
