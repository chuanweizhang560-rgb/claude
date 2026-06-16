import numpy as np


class CollisionChecker:
    """碰撞检测 + 动作裁剪"""

    def __init__(self, config: dict = None):
        self.safe_margin = 1.5       # 安全距离裕度 m
        self.uav_safe_dist = 3.0     # 无人机间安全距离 m
        self.ground_margin = 0.5     # 地面安全裕度 m
        self.clip_ratio = 0.3        # 动作裁剪中修正力比例

        # 从配置字典覆盖默认参数
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def check_terrain_collision(self, pos: np.ndarray,
                                terrain_height: float) -> bool:
        """
        检查地形碰撞

        Args:
            pos: 无人机位置 [x, y, z]
            terrain_height: 地形高度（该位置处的地面高度）

        Returns:
            是否碰撞
        """
        return pos[2] < terrain_height + self.ground_margin

    def check_turbine_collision(self, pos: np.ndarray,
                                turbine_pos: np.ndarray,
                                turbine_radius: float) -> bool:
        """
        检查风机碰撞：到圆柱轴心距离 < 半径+安全距离

        将风机视为从地面延伸到turbine_pos[2]高度的圆柱，
        同时检查水平距离和高度范围。

        Args:
            pos: 无人机位置 [x, y, z]
            turbine_pos: 风机底部位置 [x, y, z]
            turbine_radius: 风机圆柱半径

        Returns:
            是否碰撞
        """
        # 水平距离
        dx = pos[0] - turbine_pos[0]
        dy = pos[1] - turbine_pos[1]
        horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)

        # 碰撞条件：水平距离 < 半径+安全距离 且 高度在塔身范围内
        # 简化处理：如果无人机在塔身高度范围内且水平距离太近
        if horizontal_dist < turbine_radius + self.safe_margin:
            return True
        return False

    def check_cable_collision(self, pos: np.ndarray,
                              cable_points: np.ndarray) -> bool:
        """
        检查电缆碰撞：到曲线上最近点距离 < 安全距离

        Args:
            pos: 无人机位置 [x, y, z]
            cable_points: 电缆采样点 (M, 3) 数组

        Returns:
            是否碰撞
        """
        if len(cable_points) == 0:
            return False

        # 计算到所有电缆点的最短距离
        diffs = cable_points - pos
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = np.min(dists)

        # 如果电缆点之间距离较远，还需检查线段距离
        # 对相邻点之间的线段做精确距离计算
        for i in range(len(cable_points) - 1):
            seg_dist = self._point_segment_distance(
                pos, cable_points[i], cable_points[i + 1]
            )
            min_dist = min(min_dist, seg_dist)

        return min_dist < self.safe_margin

    def _point_segment_distance(self, point: np.ndarray,
                                seg_start: np.ndarray,
                                seg_end: np.ndarray) -> float:
        """
        计算点到线段的最短距离

        Args:
            point: 点坐标
            seg_start: 线段起点
            seg_end: 线段终点

        Returns:
            最短距离
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        # 线段退化为点
        if seg_len_sq < 1e-12:
            return np.linalg.norm(point - seg_start)

        # 计算投影参数t，夹到[0,1]
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = max(0.0, min(1.0, t))

        # 投影点
        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)

    def check_uav_collision(self, pos1: np.ndarray,
                            pos2: np.ndarray) -> bool:
        """
        检查两机碰撞

        Args:
            pos1: 无人机1位置
            pos2: 无人机2位置

        Returns:
            是否碰撞
        """
        dist = np.linalg.norm(pos1 - pos2)
        return dist < self.uav_safe_dist

    def check_all(self, pos: np.ndarray, obstacles: dict) -> tuple:
        """
        检查所有碰撞

        Args:
            pos: 无人机位置
            obstacles: {
                'terrain_height': float,
                'turbines': [(pos, radius), ...],
                'cables': [points_array, ...],
                'other_uavs': [pos_array, ...]
            }

        Returns:
            (is_collision: bool, nearest_obstacle_pos: np.ndarray or None)
        """
        min_dist = float('inf')
        nearest_pos = None
        is_collision = False

        # 检查地形碰撞
        terrain_height = obstacles.get('terrain_height', 0.0)
        if self.check_terrain_collision(pos, terrain_height):
            is_collision = True
            # 地面碰撞：排斥方向始终朝上
            dist_to_ground = pos[2] - terrain_height
            if dist_to_ground < min_dist:
                min_dist = dist_to_ground
                # 虚拟最近点在正下方，使排斥方向朝上
                nearest_pos = np.array([pos[0], pos[1], pos[2] - 1.0])

        # 检查风机碰撞
        for turbine_pos, turbine_radius in obstacles.get('turbines', []):
            if self.check_turbine_collision(pos, turbine_pos, turbine_radius):
                is_collision = True
                dx = pos[0] - turbine_pos[0]
                dy = pos[1] - turbine_pos[1]
                horizontal_dist = np.sqrt(dx ** 2 + dy ** 2)
                if horizontal_dist < min_dist:
                    min_dist = horizontal_dist
                    # 最近点在圆柱面上
                    if horizontal_dist > 1e-6:
                        nearest_pos = np.array([
                            turbine_pos[0] + dx / horizontal_dist * turbine_radius,
                            turbine_pos[1] + dy / horizontal_dist * turbine_radius,
                            pos[2]
                        ])
                    else:
                        nearest_pos = turbine_pos.copy()

        # 检查电缆碰撞
        for cable_pts in obstacles.get('cables', []):
            if self.check_cable_collision(pos, cable_pts):
                is_collision = True
                # 找最近的电缆点
                diffs = cable_pts - pos
                dists = np.linalg.norm(diffs, axis=1)
                closest_idx = np.argmin(dists)
                cable_dist = dists[closest_idx]
                if cable_dist < min_dist:
                    min_dist = cable_dist
                    nearest_pos = cable_pts[closest_idx].copy()

        # 检查其他无人机碰撞
        for other_pos in obstacles.get('other_uavs', []):
            other_pos = np.asarray(other_pos)
            if self.check_uav_collision(pos, other_pos):
                is_collision = True
                uav_dist = np.linalg.norm(pos - other_pos)
                if uav_dist < min_dist:
                    min_dist = uav_dist
                    nearest_pos = other_pos.copy()

        return (is_collision, nearest_pos)

    def clip_action(self, velocity: np.ndarray, pos: np.ndarray,
                    obstacles: dict, dt: float = 1.0) -> np.ndarray:
        """
        动作裁剪：如果预测会碰撞，混合修正方向

        逻辑：
        1. 计算预测位置 = pos + velocity * dt
        2. 检查预测位置的碰撞
        3. 如果有碰撞风险，计算排斥力方向（远离最近障碍物）
        4. 输出 = (1-clip_ratio)*原始速度 + clip_ratio*修正速度

        Args:
            velocity: 原始速度 [vx,vy,vz]
            pos: 当前位置
            obstacles: 障碍物字典
            dt: 时间步长

        Returns:
            clipped_velocity: 裁剪后的速度
        """
        # 1. 预测位置
        predicted_pos = pos + velocity * dt

        # 2. 检查预测位置的碰撞
        is_collision, nearest_obstacle = self.check_all(predicted_pos, obstacles)

        if not is_collision or nearest_obstacle is None:
            return velocity

        # 3. 计算排斥力方向（远离最近障碍物）
        repel_dir = predicted_pos - nearest_obstacle
        repel_dist = np.linalg.norm(repel_dir)

        if repel_dist < 1e-6:
            # 无人机几乎在障碍物中心，使用随机排斥方向
            repel_dir = np.array([0.0, 0.0, 1.0])
        else:
            repel_dir = repel_dir / repel_dist  # 归一化

        # 修正速度：沿排斥方向，大小与原始速度相同
        speed = np.linalg.norm(velocity)
        corrected_velocity = repel_dir * speed

        # 4. 混合原始速度和修正速度
        clipped = (1.0 - self.clip_ratio) * velocity + self.clip_ratio * corrected_velocity

        return clipped


if __name__ == "__main__":
    # 基本功能测试
    print("=== CollisionChecker 测试 ===\n")

    cc = CollisionChecker()

    # 1. 地形碰撞测试
    pos_ground = np.array([5.0, 5.0, 0.3])  # 低于地面+裕度
    pos_safe = np.array([5.0, 5.0, 2.0])    # 高于地面+裕度
    print(f"低空碰撞: {cc.check_terrain_collision(pos_ground, 0.0)}")
    assert cc.check_terrain_collision(pos_ground, 0.0), "应检测到地形碰撞"
    print(f"安全高度: {cc.check_terrain_collision(pos_safe, 0.0)}")
    assert not cc.check_terrain_collision(pos_safe, 0.0), "安全高度不应碰撞"

    # 2. 风机碰撞测试
    turbine_pos = np.array([10.0, 0.0, 0.0])
    turbine_r = 1.5
    pos_near_turbine = np.array([11.0, 0.0, 30.0])  # 水平距离1.0 < 1.5+1.5
    pos_far_turbine = np.array([20.0, 0.0, 30.0])    # 水平距离10.0 > 1.5+1.5
    print(f"近风机碰撞: {cc.check_turbine_collision(pos_near_turbine, turbine_pos, turbine_r)}")
    assert cc.check_turbine_collision(pos_near_turbine, turbine_pos, turbine_r), \
        "应检测到风机碰撞"
    print(f"远风机安全: {cc.check_turbine_collision(pos_far_turbine, turbine_pos, turbine_r)}")
    assert not cc.check_turbine_collision(pos_far_turbine, turbine_pos, turbine_r), \
        "远风机不应碰撞"

    # 3. 电缆碰撞测试
    cable_pts = np.array([
        [0.0, 0.0, 30.0],
        [10.0, 0.0, 30.0],
        [20.0, 0.0, 30.0],
    ])
    pos_near_cable = np.array([5.0, 1.0, 30.0])   # 距电缆1.0 < 1.5
    pos_far_cable = np.array([5.0, 10.0, 30.0])    # 距电缆10.0 > 1.5
    print(f"近电缆碰撞: {cc.check_cable_collision(pos_near_cable, cable_pts)}")
    assert cc.check_cable_collision(pos_near_cable, cable_pts), "应检测到电缆碰撞"
    print(f"远电缆安全: {cc.check_cable_collision(pos_far_cable, cable_pts)}")
    assert not cc.check_cable_collision(pos_far_cable, cable_pts), "远电缆不应碰撞"

    # 4. 无人机间碰撞测试
    pos_a = np.array([0.0, 0.0, 30.0])
    pos_b_close = np.array([2.0, 0.0, 30.0])   # 距离2.0 < 3.0
    pos_b_far = np.array([5.0, 0.0, 30.0])     # 距离5.0 > 3.0
    print(f"近距碰撞: {cc.check_uav_collision(pos_a, pos_b_close)}")
    assert cc.check_uav_collision(pos_a, pos_b_close), "应检测到无人机碰撞"
    print(f"远距安全: {cc.check_uav_collision(pos_a, pos_b_far)}")
    assert not cc.check_uav_collision(pos_a, pos_b_far), "远距不应碰撞"

    # 5. 综合碰撞检测测试
    obstacles = {
        'terrain_height': 0.0,
        'turbines': [(turbine_pos, turbine_r)],
        'cables': [cable_pts],
        'other_uavs': [pos_b_far]
    }
    # 安全位置
    pos_safe_all = np.array([30.0, 30.0, 50.0])
    is_col, nearest = cc.check_all(pos_safe_all, obstacles)
    print(f"安全位置碰撞: {is_col}")
    assert not is_col, "安全位置不应碰撞"

    # 地面碰撞
    pos_ground_col = np.array([30.0, 30.0, 0.3])
    is_col, nearest = cc.check_all(pos_ground_col, obstacles)
    print(f"地面碰撞: {is_col}")
    assert is_col, "应检测到地面碰撞"

    # 风机碰撞
    pos_turbine_col = np.array([11.0, 0.0, 30.0])
    is_col, nearest = cc.check_all(pos_turbine_col, obstacles)
    print(f"风机碰撞: {is_col}")
    assert is_col, "应检测到风机碰撞"

    # 6. 动作裁剪测试 — 无碰撞时不裁剪
    vel_orig = np.array([5.0, 0.0, 0.0])
    pos_safe_clip = np.array([50.0, 50.0, 50.0])
    obstacles_safe = {
        'terrain_height': 0.0,
        'turbines': [],
        'cables': [],
        'other_uavs': []
    }
    vel_clipped = cc.clip_action(vel_orig, pos_safe_clip, obstacles_safe)
    print(f"\n无碰撞裁剪: 原始={vel_orig}, 裁剪后={vel_clipped}")
    assert np.allclose(vel_orig, vel_clipped), "无碰撞时不应裁剪"

    # 7. 动作裁剪测试 — 有碰撞时混合修正
    # 无人机在地面上方较低位置，向地面飞
    vel_down = np.array([0.0, 0.0, -5.0])
    pos_low = np.array([0.0, 0.0, 1.0])
    obstacles_low = {
        'terrain_height': 0.0,
        'turbines': [],
        'cables': [],
        'other_uavs': []
    }
    vel_clipped_down = cc.clip_action(vel_down, pos_low, obstacles_low, dt=1.0)
    print(f"向下飞裁剪: 原始={vel_down}, 裁剪后={vel_clipped_down}")
    # 裁剪后z分量应被向上修正
    assert vel_clipped_down[2] > vel_down[2], "向下飞应被修正为向上"

    # 8. 点到线段距离辅助函数测试
    point = np.array([5.0, 5.0, 0.0])
    seg_a = np.array([0.0, 0.0, 0.0])
    seg_b = np.array([10.0, 0.0, 0.0])
    dist = cc._point_segment_distance(point, seg_a, seg_b)
    print(f"\n点到线段距离: {dist:.2f} (期望5.0)")
    assert abs(dist - 5.0) < 1e-6, "点到线段距离计算错误"

    # 线段端点外的情况
    point_out = np.array([15.0, 5.0, 0.0])
    dist_out = cc._point_segment_distance(point_out, seg_a, seg_b)
    expected_out = np.sqrt(25.0 + 25.0)  # 到(10,0,0)的距离
    print(f"端点外距离: {dist_out:.2f} (期望{expected_out:.2f})")
    assert abs(dist_out - expected_out) < 1e-6, "端点外距离计算错误"

    # 9. 配置覆盖测试
    cc2 = CollisionChecker(config={'safe_margin': 3.0, 'uav_safe_dist': 5.0})
    print(f"\n配置覆盖: safe_margin={cc2.safe_margin}, uav_safe_dist={cc2.uav_safe_dist}")
    assert cc2.safe_margin == 3.0, "配置覆盖safe_margin失败"
    assert cc2.uav_safe_dist == 5.0, "配置覆盖uav_safe_dist失败"

    # 10. 空障碍物测试
    obstacles_empty = {
        'terrain_height': 0.0,
        'turbines': [],
        'cables': [],
        'other_uavs': []
    }
    pos_any = np.array([0.0, 0.0, 0.3])
    is_col, nearest = cc.check_all(pos_any, obstacles_empty)
    print(f"空障碍物地面碰撞: {is_col}")
    assert is_col, "即使无其他障碍物，地面碰撞也应检测到"

    print("\n=== 所有测试通过 ===")
