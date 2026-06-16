import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.controllers.base_controller import RuleController


class ReturnHomeController(RuleController):
    """
    返航引导：飞回基站

    直线飞回基站，到达基站附近后减速悬停
    """

    def __init__(self, config: dict = None):
        self.return_speed = 4.0          # 返航速度 m/s
        self.slow_radius = 10.0         # 减速距离
        self.hover_height = 5.0         # 悬停高度
        self.dt = 0.05

        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_guided_velocity(self, state: dict) -> np.ndarray:
        pos = np.asarray(state['pos'], dtype=np.float64)
        vel = np.asarray(state['vel'], dtype=np.float64)
        yaw = state.get('yaw', 0.0)
        base_pos = np.asarray(state.get('base_pos', [0.0, 0.0, 0.0]), dtype=np.float64)

        # 1. 计算到基站的方向和距离
        to_base = base_pos - pos
        dist = np.linalg.norm(to_base)

        if dist < 1.0:
            # 已到达基站，悬停
            # 调整高度到悬停高度
            if abs(pos[2] - self.hover_height) > 0.5:
                vz = (self.hover_height - pos[2]) * 0.5
            else:
                vz = 0.0
            return np.array([0.0, 0.0, vz, 0.0])

        # 2. 计算速度
        direction = to_base / dist

        if dist > self.slow_radius:
            # 全速返航
            speed = self.return_speed
        else:
            # 减速
            speed = self.return_speed * (dist / self.slow_radius)

        vel_cmd = direction * speed

        # 3. 到达基站附近时调整高度
        if dist < self.slow_radius:
            # 同时调整到悬停高度
            height_diff = self.hover_height - pos[2]
            vel_cmd[2] = np.clip(height_diff * 0.5, -2.0, 2.0)

        # 4. yaw指向基站方向
        target_yaw = self._yaw_to_target(pos, base_pos)
        yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)

        return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])


if __name__ == "__main__":
    print("=== ReturnHomeController 测试 ===\n")

    ctrl = ReturnHomeController()
    base_pos = np.array([0.0, 0.0, 0.0])

    # 测试1: 远处返航
    state_far = {
        'pos': np.array([50.0, 50.0, 30.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'base_pos': base_pos,
    }
    vel_far = ctrl.get_guided_velocity(state_far)
    print(f"远处返航: vel={vel_far}, 应朝基站飞")
    assert vel_far[0] < 0 and vel_far[1] < 0, "应朝基站飞"

    # 测试2: 近处减速
    state_near = {
        'pos': np.array([5.0, 0.0, 10.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'base_pos': base_pos,
    }
    vel_near = ctrl.get_guided_velocity(state_near)
    speed_near = np.linalg.norm(vel_near[:3])
    print(f"近处减速: vel={vel_near}, 速度={speed_near:.2f} (应<4.0)")
    assert speed_near < 4.0, "近处应减速"

    # 测试3: 到达悬停
    state_at = {
        'pos': np.array([0.5, 0.0, 5.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'base_pos': base_pos,
    }
    vel_at = ctrl.get_guided_velocity(state_at)
    print(f"到达悬停: vel={vel_at}, 水平速度应≈0")
    assert np.linalg.norm(vel_at[:2]) < 0.5, "到达后应悬停"

    # 测试4: 高度调整
    state_high = {
        'pos': np.array([0.5, 0.0, 20.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'base_pos': base_pos,
    }
    vel_high = ctrl.get_guided_velocity(state_high)
    print(f"高处调整: vz={vel_high[2]:.2f}, 应<0 (下降)")
    assert vel_high[2] < 0, "高处应下降"

    print("\n=== 所有测试通过 ===")
