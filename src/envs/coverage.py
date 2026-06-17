import numpy as np


class CoverageEvaluator:
    """距离阈值覆盖评估"""

    def __init__(self, config: dict = None):
        self.turbine_radius = 1.5    # 风机圆柱半径 m
        self.turbine_height = 70.0   # 风机高度 m
        self.coverage_dist_turbine = 12.0  # 风机覆盖判定距离 m
        self.coverage_dist_cable = 5.0     # 电缆覆盖判定距离 m

        # 从配置字典覆盖默认参数
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def compute_turbine_coverage(self, uav_pos: np.ndarray,
                                 turbine_pos: np.ndarray) -> float:
        """
        计算单个无人机的风机覆盖值

        用分段评估：将风机塔身沿高度分成N段，
        无人机到某段距离<coverage_dist则该段已覆盖

        Args:
            uav_pos: 无人机位置 [x, y, z]
            turbine_pos: 风机底部位置 [x, y, z]（塔底）

        Returns:
            coverage: [0, 1] 覆盖进度
        """
        return self.compute_turbine_segment_coverage(uav_pos, turbine_pos).mean()

    def compute_turbine_segment_coverage(self, uav_pos: np.ndarray,
                                           turbine_pos: np.ndarray) -> np.ndarray:
        """
        计算风机每段的覆盖状态

        Returns:
            covered: (n_segments,) bool数组，每段是否被覆盖
        """
        n_segments = 20
        segment_heights = np.linspace(0, self.turbine_height, n_segments + 1)
        segment_centers = (segment_heights[:-1] + segment_heights[1:]) / 2.0

        covered = np.zeros(n_segments, dtype=bool)
        for j, h in enumerate(segment_centers):
            segment_pos = np.array([
                turbine_pos[0],
                turbine_pos[1],
                turbine_pos[2] + h
            ])
            dist = np.linalg.norm(uav_pos - segment_pos)
            if dist < self.coverage_dist_turbine:
                covered[j] = True

        return covered

    def compute_cable_coverage(self, uav_pos: np.ndarray,
                               cable_points: np.ndarray) -> float:
        """
        计算单个无人机的电缆覆盖值

        将电缆分成N段，无人机到某段距离<coverage_dist则该段已覆盖

        Args:
            uav_pos: 无人机位置 [x, y, z]
            cable_points: 电缆采样点 (M, 3) 数组

        Returns:
            coverage: [0, 1] 覆盖进度
        """
        return self.compute_cable_segment_coverage(uav_pos, cable_points).mean()

    def compute_cable_segment_coverage(self, uav_pos: np.ndarray,
                                         cable_points: np.ndarray) -> np.ndarray:
        """
        计算电缆每段的覆盖状态

        Returns:
            covered: (n_segments,) bool数组，每段是否被覆盖
        """
        if len(cable_points) < 2:
            return np.array([], dtype=bool)

        n_segments = len(cable_points) - 1
        covered = np.zeros(n_segments, dtype=bool)

        for i in range(n_segments):
            center = (cable_points[i] + cable_points[i + 1]) / 2.0
            dist = np.linalg.norm(uav_pos - center)
            if dist < self.coverage_dist_cable:
                covered[i] = True

        return covered

    def check_turbine_covered(self, coverage: float,
                              threshold: float = 0.85) -> bool:
        """
        风机是否已完全覆盖

        Args:
            coverage: 覆盖进度 [0, 1]
            threshold: 覆盖完成阈值

        Returns:
            是否已覆盖
        """
        return coverage >= threshold

    def check_cable_covered(self, coverage: float,
                            threshold: float = 0.85) -> bool:
        """
        电缆是否已完全覆盖

        Args:
            coverage: 覆盖进度 [0, 1]
            threshold: 覆盖完成阈值

        Returns:
            是否已覆盖
        """
        return coverage >= threshold


if __name__ == "__main__":
    # 基本功能测试
    print("=== CoverageEvaluator 测试 ===\n")

    ce = CoverageEvaluator()

    # 1. 风机覆盖 — 无人机远离风机
    uav_far = np.array([100.0, 0.0, 35.0])
    turbine_pos = np.array([0.0, 0.0, 0.0])
    cov_far = ce.compute_turbine_coverage(uav_far, turbine_pos)
    print(f"远离风机时覆盖: {cov_far:.2f}")
    assert cov_far == 0.0, "远离风机应无覆盖"

    # 2. 风机覆盖 — 无人机在塔顶附近
    uav_top = np.array([0.0, 0.0, 70.0])
    cov_top = ce.compute_turbine_coverage(uav_top, turbine_pos)
    print(f"塔顶附近覆盖: {cov_top:.2f}")
    assert cov_top > 0.0, "塔顶附近应有覆盖"
    assert cov_top < 1.0, "塔顶附近不应完全覆盖"

    # 3. 风机覆盖 — 无人机在塔中间高度，距离很近
    uav_mid = np.array([0.0, 0.0, 35.0])
    cov_mid = ce.compute_turbine_coverage(uav_mid, turbine_pos)
    print(f"塔中间高度覆盖: {cov_mid:.2f}")
    assert cov_mid > 0.0, "塔中间应有覆盖"

    # 4. 风机覆盖 — 无人机紧贴塔身（全覆盖）
    # 覆盖距离12m，如果无人机沿塔身飞行且距离<12m，可覆盖多段
    # 在塔底正上方0距离，覆盖距离12m内的高度段
    uav_close = np.array([0.0, 0.0, 0.0])
    cov_close = ce.compute_turbine_coverage(uav_close, turbine_pos)
    print(f"塔底紧贴覆盖: {cov_close:.2f}")
    # 塔底0m处，覆盖距离12m，可覆盖0~12m高度的段
    # 20段，每段3.5m，0~12m约覆盖前3-4段
    assert cov_close > 0.0, "紧贴塔底应有覆盖"

    # 5. 风机覆盖判定
    assert ce.check_turbine_covered(0.9), "0.9应判定为已覆盖"
    assert not ce.check_turbine_covered(0.8), "0.8不应判定为已覆盖"
    print("风机覆盖判定测试通过")

    # 6. 电缆覆盖 — 无人机远离电缆
    cable_pts = np.array([
        [0.0, 0.0, 30.0],
        [10.0, 0.0, 30.0],
        [20.0, 0.0, 30.0],
        [30.0, 0.0, 30.0],
    ])
    uav_cable_far = np.array([100.0, 100.0, 30.0])
    cov_cable_far = ce.compute_cable_coverage(uav_cable_far, cable_pts)
    print(f"远离电缆时覆盖: {cov_cable_far:.2f}")
    assert cov_cable_far == 0.0, "远离电缆应无覆盖"

    # 7. 电缆覆盖 — 无人机在电缆中点附近
    uav_cable_mid = np.array([15.0, 0.0, 30.0])
    cov_cable_mid = ce.compute_cable_coverage(uav_cable_mid, cable_pts)
    print(f"电缆中点附近覆盖: {cov_cable_mid:.2f}")
    assert cov_cable_mid > 0.0, "电缆中点附近应有覆盖"

    # 8. 电缆覆盖 — 无人机紧贴电缆（全覆盖）
    uav_cable_close = np.array([15.0, 0.0, 30.0])
    cov_cable_close = ce.compute_cable_coverage(uav_cable_close, cable_pts)
    print(f"紧贴电缆覆盖: {cov_cable_close:.2f}")
    # 3段电缆，中点在5,15,25处，距离0,0,10 -> 覆盖2段
    assert cov_cable_close > 0.0, "紧贴电缆应有覆盖"

    # 9. 电缆覆盖判定
    assert ce.check_cable_covered(0.9), "0.9应判定为已覆盖"
    assert not ce.check_cable_covered(0.8), "0.8不应判定为已覆盖"
    print("电缆覆盖判定测试通过")

    # 10. 空电缆点测试
    empty_cable = np.array([[0.0, 0.0, 0.0]])
    cov_empty = ce.compute_cable_coverage(uav_cable_mid, empty_cable)
    print(f"单点电缆覆盖: {cov_empty:.2f}")
    assert cov_empty == 0.0, "单点电缆应无覆盖"

    # 11. 配置覆盖测试
    ce2 = CoverageEvaluator(config={'turbine_height': 100.0,
                                    'coverage_dist_turbine': 20.0})
    print(f"\n配置覆盖后塔高: {ce2.turbine_height}, 覆盖距离: {ce2.coverage_dist_turbine}")
    assert ce2.turbine_height == 100.0, "配置覆盖塔高失败"
    assert ce2.coverage_dist_turbine == 20.0, "配置覆盖距离失败"

    print("\n=== 所有测试通过 ===")
