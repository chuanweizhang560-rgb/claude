import numpy as np


class RuleController:
    """规则控制器基类"""

    def get_guided_velocity(self, state: dict) -> np.ndarray:
        """
        Args:
            state: {
                'pos': [x,y,z],
                'vel': [vx,vy,vz],
                'yaw': float,
                'target': dict,
                'role': str,  # 'primary' or 'assistant'
                'battery': float,
                'entropy_map': np.ndarray,  # 8扇区熵值
                'base_pos': [x,y,z],
            }
        Returns:
            guided_velocity: [vx, vy, vz, yaw_rate] 引导速度
        """
        raise NotImplementedError

    def _vec2d(self, v):
        """取xy分量"""
        return np.array([v[0], v[1]])

    def _normalize_angle(self, angle):
        """角度归一化到 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _yaw_to_target(self, pos, target_pos):
        """计算指向目标点的yaw角"""
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        return np.arctan2(dy, dx)

    def _yaw_rate_to_target(self, current_yaw, target_yaw, dt=0.05, max_rate=1.0):
        """计算到达目标yaw所需的yaw_rate"""
        diff = self._normalize_angle(target_yaw - current_yaw)
        rate = diff / dt
        return np.clip(rate, -max_rate, max_rate)
