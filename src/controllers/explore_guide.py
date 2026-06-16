import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.controllers.base_controller import RuleController


class ExploreGuideController(RuleController):
    """
    探索引导：往高熵区域飞

    选择8扇区中熵值最高的方向前进
    """

    def __init__(self, config: dict = None):
        self.explore_speed = 2.0         # 探索速度 m/s
        self.explore_height = 25.0       # 探索飞行高度
        self.min_entropy_threshold = 0.3 # 最低熵阈值，低于此不探索
        self.dt = 0.05

        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_guided_velocity(self, state: dict) -> np.ndarray:
        pos = np.asarray(state['pos'], dtype=np.float64)
        vel = np.asarray(state['vel'], dtype=np.float64)
        yaw = state.get('yaw', 0.0)
        entropy_map = state.get('entropy_map', np.ones(8) * 0.5)

        # 1. 找到熵最高的扇区
        max_sector = np.argmax(entropy_map)
        max_entropy = entropy_map[max_sector]

        # 如果所有扇区熵都很低，不需要探索
        if max_entropy < self.min_entropy_threshold:
            # 悬停
            return np.array([0.0, 0.0, 0.0, 0.0])

        # 2. 计算该扇区方向（扇区i对应角度 i*45°，扇区0=正东方向）
        # 扇区0=0°, 1=45°, 2=90°, ..., 7=315°
        sector_angle = max_sector * (2 * np.pi / 8)

        # 3. 水平速度指向最高熵方向
        vx = self.explore_speed * np.cos(sector_angle)
        vy = self.explore_speed * np.sin(sector_angle)

        # 4. 高度控制：如果低于探索高度，先爬升
        if pos[2] < self.explore_height - 2.0:
            vz = 2.0  # 爬升
        elif pos[2] > self.explore_height + 2.0:
            vz = -1.0  # 下降
        else:
            vz = 0.0

        # 5. yaw指向飞行方向
        target_yaw = sector_angle
        yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)

        return np.array([vx, vy, vz, yaw_rate])


if __name__ == "__main__":
    print("=== ExploreGuideController 测试 ===\n")

    ctrl = ExploreGuideController()

    # 测试1: 北方熵最高（扇区2=90°）
    entropy_north = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    state1 = {
        'pos': np.array([0.0, 0.0, 25.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'entropy_map': entropy_north,
    }
    vel1 = ctrl.get_guided_velocity(state1)
    print(f"北方高熵: vel={vel1}, vy应>0 (向北飞)")
    assert vel1[1] > 0, "应向北飞"

    # 测试2: 西方熵最高（扇区4=180°）
    entropy_west = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    state2 = {
        'pos': np.array([0.0, 0.0, 25.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'entropy_map': entropy_west,
    }
    vel2 = ctrl.get_guided_velocity(state2)
    print(f"西方高熵: vel={vel2}, vx应<0 (向西飞)")
    assert vel2[0] < 0, "应向西飞"

    # 测试3: 低空应先爬升
    state3 = {
        'pos': np.array([0.0, 0.0, 10.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'entropy_map': entropy_north,
    }
    vel3 = ctrl.get_guided_velocity(state3)
    print(f"低空: vz={vel3[2]:.2f}, 应>0 (爬升)")
    assert vel3[2] > 0, "低空应爬升"

    # 测试4: 所有熵低时悬停
    entropy_low = np.ones(8) * 0.1
    state4 = {
        'pos': np.array([0.0, 0.0, 25.0]),
        'vel': np.zeros(3),
        'yaw': 0.0,
        'entropy_map': entropy_low,
    }
    vel4 = ctrl.get_guided_velocity(state4)
    print(f"低熵: vel={vel4}, 应全零 (悬停)")
    assert np.allclose(vel4, 0.0), "低熵应悬停"

    print("\n=== 所有测试通过 ===")
