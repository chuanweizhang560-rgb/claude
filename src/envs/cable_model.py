import numpy as np


class CableModel:
    """抛物线近似悬链线电缆模型"""

    def __init__(self, start: np.ndarray, end: np.ndarray, sag: float = 4.0):
        """
        Args:
            start: 电缆起点 [x,y,z]
            end: 电缆终点 [x,y,z]
            sag: 最大垂度 m
        """
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.sag = sag
        # 直线方向向量
        self.direction = self.end - self.start
        # 水平跨度（用于计算垂度方向）
        self.horizontal = np.array([self.direction[0], self.direction[1], 0.0])
        self.horizontal_length = np.linalg.norm(self.horizontal)

    def get_point(self, t: float) -> np.ndarray:
        """
        获取电缆上t参数处的3D点

        Args:
            t: 参数 [0, 1]，0=起点，1=终点

        Returns:
            point: [x, y, z]
        """
        # 沿直线插值
        point = self.start + t * self.direction
        # z方向叠加抛物线下垂: sag * 4 * t * (1-t)
        # 抛物线在 t=0 和 t=1 时为0，在 t=0.5 时达到最大值 sag
        point[2] -= self.sag * 4.0 * t * (1.0 - t)
        return point

    def get_points(self, n: int = 50) -> np.ndarray:
        """获取电缆上n个等分点，返回 (n, 3) 数组"""
        ts = np.linspace(0.0, 1.0, n)
        # 向量化计算所有点
        points = np.zeros((n, 3))
        for i, t in enumerate(ts):
            points[i] = self.get_point(t)
        return points

    def get_nearest_point(self, pos: np.ndarray) -> tuple:
        """
        找到电缆上距离pos最近的点

        Args:
            pos: [x,y,z]

        Returns:
            (nearest_point, distance, t_param)
        """
        pos = np.asarray(pos, dtype=np.float64)
        # 用密集采样 + 黄金分割精搜索
        # 先粗搜索
        n_coarse = 200
        ts_coarse = np.linspace(0.0, 1.0, n_coarse)
        min_dist = np.inf
        best_t = 0.0
        for t in ts_coarse:
            p = self.get_point(t)
            d = np.linalg.norm(p - pos)
            if d < min_dist:
                min_dist = d
                best_t = t

        # 在最佳t附近精搜索（黄金分割法）
        lo = max(0.0, best_t - 1.0 / n_coarse)
        hi = min(1.0, best_t + 1.0 / n_coarse)
        golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(50):  # 50次迭代足够精确
            t1 = hi - golden_ratio * (hi - lo)
            t2 = lo + golden_ratio * (hi - lo)
            d1 = np.linalg.norm(self.get_point(t1) - pos)
            d2 = np.linalg.norm(self.get_point(t2) - pos)
            if d1 < d2:
                hi = t2
            else:
                lo = t1

        best_t = (lo + hi) / 2.0
        nearest_point = self.get_point(best_t)
        distance = np.linalg.norm(nearest_point - pos)
        return (nearest_point, distance, best_t)

    def get_tangent(self, t: float) -> np.ndarray:
        """获取t参数处的切线方向（单位向量）"""
        # 数值微分求切线
        dt = 1e-6
        t0 = max(0.0, t - dt)
        t1 = min(1.0, t + dt)
        p0 = self.get_point(t0)
        p1 = self.get_point(t1)
        tangent = p1 - p0
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            # 退化为零向量时返回默认方向
            return self.direction / (np.linalg.norm(self.direction) + 1e-12)
        return tangent / norm

    def get_length(self) -> float:
        """计算电缆近似长度"""
        # 用密集采样点求和近似弧长
        n = 500
        points = self.get_points(n)
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(segment_lengths))


if __name__ == "__main__":
    print("=== CableModel 测试 ===")

    # 测试1: 基本创建和get_point
    start = np.array([0.0, 0.0, 30.0])
    end = np.array([100.0, 0.0, 30.0])
    cable = CableModel(start, end, sag=4.0)

    # 起点
    p0 = cable.get_point(0.0)
    print(f"t=0 处点: {p0}, 期望: {start}")
    assert np.allclose(p0, start), "起点不匹配"

    # 终点
    p1 = cable.get_point(1.0)
    print(f"t=1 处点: {p1}, 期望: {end}")
    assert np.allclose(p1, end), "终点不匹配"

    # 中点（垂度最大）
    p_mid = cable.get_point(0.5)
    expected_z = 30.0 - 4.0  # z = 30 - sag
    print(f"t=0.5 处点: {p_mid}, z期望: {expected_z}, 实际z: {p_mid[2]:.4f}")
    assert abs(p_mid[2] - expected_z) < 0.01, "中点垂度不正确"

    # 测试2: get_points
    points = cable.get_points(50)
    print(f"get_points(50) 形状: {points.shape}")
    assert points.shape == (50, 3), "形状不正确"

    # 测试3: get_nearest_point
    # 在电缆正下方找最近点
    query = np.array([50.0, 0.0, 20.0])
    nearest, dist, t_param = cable.get_nearest_point(query)
    print(f"查询点: {query}, 最近点: {nearest}, 距离: {dist:.4f}, t: {t_param:.4f}")
    # 最近点应该在 t≈0.5 附近（水平方向中间）
    assert abs(t_param - 0.5) < 0.05, "最近点t参数不正确"

    # 测试4: get_tangent
    tangent = cable.get_tangent(0.5)
    print(f"t=0.5 切线方向: {tangent}, 模长: {np.linalg.norm(tangent):.6f}")
    assert abs(np.linalg.norm(tangent) - 1.0) < 1e-6, "切线不是单位向量"

    # 测试5: get_length
    length = cable.get_length()
    print(f"电缆长度: {length:.4f} m (直线距离: {np.linalg.norm(end - start):.4f} m)")
    # 由于下垂，电缆长度应大于直线距离
    assert length > np.linalg.norm(end - start), "电缆长度应大于直线距离"

    # 测试6: 倾斜电缆
    start2 = np.array([0.0, 0.0, 50.0])
    end2 = np.array([80.0, 60.0, 20.0])
    cable2 = CableModel(start2, end2, sag=3.0)
    p_mid2 = cable2.get_point(0.5)
    print(f"\n倾斜电缆中点: {p_mid2}")
    length2 = cable2.get_length()
    print(f"倾斜电缆长度: {length2:.4f} m")

    print("\n=== 所有测试通过 ===")
