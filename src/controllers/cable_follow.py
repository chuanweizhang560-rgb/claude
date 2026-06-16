import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.controllers.base_controller import RuleController


class CableFollowController(RuleController):
    """
    电缆Frenet跟随控制器

    primary角色：沿电缆前进方向飞行，保持在电缆正上方
    assistant角色：沿电缆前进方向飞行，横向偏移5m
    """

    def __init__(self, config: dict = None):
        self.follow_height_offset = 3.0   # 飞行高度在电缆上方3m
        self.assistant_offset = 5.0        # assistant横向偏移m
        self.follow_speed = 2.0            # 沿电缆前进速度 m/s
        self.approach_speed = 3.0          # 接近阶段速度 m/s
        self.t_progress = 0.0              # Frenet参数进度 [0,1]
        self.dt = 0.05

        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def reset(self):
        """重置进度"""
        self.t_progress = 0.0

    def get_guided_velocity(self, state: dict) -> np.ndarray:
        pos = np.asarray(state['pos'], dtype=np.float64)
        vel = np.asarray(state['vel'], dtype=np.float64)
        yaw = state.get('yaw', 0.0)
        target = state['target']
        role = state.get('role', 'primary')

        cable_points = np.asarray(target['points'], dtype=np.float64)
        n_pts = len(cable_points)

        # 1. 找到最近的电缆点，确定当前t_progress
        diffs = cable_points - pos
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argmin(dists)
        self.t_progress = nearest_idx / max(n_pts - 1, 1)

        # 2. 计算目标点（沿电缆前进一步）
        next_idx = min(nearest_idx + max(1, int(self.follow_speed * self.dt / 2.0)), n_pts - 1)
        target_point = cable_points[next_idx].copy()

        # 3. 根据角色决定偏移
        if role == 'assistant':
            # assistant横向偏移：计算电缆切线，取法线方向偏移
            if next_idx < n_pts - 1:
                tangent = cable_points[min(next_idx + 1, n_pts - 1)] - cable_points[next_idx]
            else:
                tangent = cable_points[next_idx] - cable_points[max(next_idx - 1, 0)]
            tangent_xy = self._vec2d(tangent)
            tang_norm = np.linalg.norm(tangent_xy)
            if tang_norm > 1e-6:
                tangent_xy = tangent_xy / tang_norm
            # 法线方向（切线逆时针旋转90度）
            normal_xy = np.array([-tangent_xy[1], tangent_xy[0]])
            target_point[0] += normal_xy[0] * self.assistant_offset
            target_point[1] += normal_xy[1] * self.assistant_offset

        # 飞行高度在电缆上方
        target_point[2] = cable_points[next_idx][2] + self.follow_height_offset

        # 4. 计算到目标点的方向和距离
        to_target = target_point - pos
        dist = np.linalg.norm(to_target)

        if dist < 0.5:
            # 已到达目标点附近，沿电缆切线方向前进
            if next_idx < n_pts - 1:
                tangent = cable_points[min(next_idx + 1, n_pts - 1)] - cable_points[next_idx]
            else:
                tangent = cable_points[next_idx] - cable_points[max(next_idx - 1, 0)]
            tang_norm = np.linalg.norm(tangent)
            if tang_norm > 1e-6:
                vel_cmd = tangent / tang_norm * self.follow_speed
            else:
                vel_cmd = np.zeros(3)
        else:
            # 飞向目标点
            speed = min(self.approach_speed, dist * 2.0)
            vel_cmd = to_target / dist * speed

        # 5. yaw指向电缆切线方向（primary）或偏置方向（assistant）
        if next_idx < n_pts - 1:
            tangent = cable_points[min(next_idx + 1, n_pts - 1)] - cable_points[next_idx]
        else:
            tangent = cable_points[next_idx] - cable_points[max(next_idx - 1, 0)]

        if role == 'assistant':
            # assistant看向电缆（偏移方向的反方向）
            look_target = cable_points[next_idx]
        else:
            # primary沿电缆方向看
            look_target = pos + tangent

        target_yaw = self._yaw_to_target(pos, look_target)
        yaw_rate = self._yaw_rate_to_target(yaw, target_yaw, self.dt)

        return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])


if __name__ == "__main__":
    print("=== CableFollowController 测试 ===\n")

    # 创建电缆：从(-30, 0, 25)到(30, 0, 25)，垂度4m
    n_pts = 50
    ts = np.linspace(0, 1, n_pts)
    cable_pts = np.zeros((n_pts, 3))
    for i, t in enumerate(ts):
        cable_pts[i] = [-30 + 60 * t, 0.0, 25 + 4 * 4 * t * (1 - t)]
    print(f"电缆: {n_pts}点, 起点{cable_pts[0]}, 终点{cable_pts[-1]}")

    ctrl = CableFollowController()

    # 测试1: primary从电缆中间开始
    state_primary = {
        'pos': np.array([0.0, 0.0, 30.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'yaw': 0.0,
        'target': {'type': 'cable', 'points': cable_pts, 'start': cable_pts[0], 'end': cable_pts[-1]},
        'role': 'primary',
    }
    vel_primary = ctrl.get_guided_velocity(state_primary)
    print(f"\nprimary at (0,0,30): vel={vel_primary}, vx>0 (前进方向)")

    # 测试2: assistant从同一位置
    ctrl2 = CableFollowController()
    state_assistant = state_primary.copy()
    state_assistant['role'] = 'assistant'
    vel_assistant = ctrl2.get_guided_velocity(state_assistant)
    print(f"assistant at (0,0,30): vel={vel_assistant}, vy偏移 (横向)")

    # 测试3: 从远处接近
    ctrl3 = CableFollowController()
    state_far = {
        'pos': np.array([0.0, 20.0, 30.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'yaw': 0.0,
        'target': {'type': 'cable', 'points': cable_pts, 'start': cable_pts[0], 'end': cable_pts[-1]},
        'role': 'primary',
    }
    vel_far = ctrl3.get_guided_velocity(state_far)
    print(f"\n远处接近 (0,20,30): vel={vel_far}, vy应朝向电缆")

    print("\n=== 测试完成 ===")
