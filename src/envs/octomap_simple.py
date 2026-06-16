import numpy as np


class SimpleOctoMap:
    """numpy实现的简化OctoMap，数学等价于真实OctoMap"""

    def __init__(self, config: dict = None):
        self.resolution = 0.5       # 体素大小 m
        self.origin = np.array([-150.0, -150.0, 0.0])  # 地图原点
        self.size = np.array([300, 300, 50])  # 地图尺寸 m
        self.occupancy_threshold = 0.7  # 占据判定阈值
        self.free_threshold = 0.3       # 空闲判定阈值
        self.clamp_min = 0.01           # 概率下限
        self.clamp_max = 0.99           # 概率上限

        # 如果有配置覆盖，先应用配置再创建网格
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # 确保origin和size是numpy数组
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)

        # 计算网格尺寸
        self.grid_size = (self.size / self.resolution).astype(int)

        # 占据概率网格，初始0.5（最大不确定性）
        self.grid = np.full(self.grid_size, 0.5, dtype=np.float32)

        # 预计算对数概率更新量
        # 观测为占据时的更新: logit(0.7) - logit(0.5)
        # 观测为空闲时的更新: logit(0.3) - logit(0.5)
        self._logit_occ_update = self._logit(0.7) - self._logit(0.5)
        self._logit_free_update = self._logit(0.3) - self._logit(0.5)

    @staticmethod
    def _logit(p: float) -> float:
        """logit变换: log(p/(1-p))"""
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """sigmoid变换: 1/(1+exp(-x))"""
        return 1.0 / (1.0 + np.exp(-x))

    def world_to_voxel(self, pos: np.ndarray) -> tuple:
        """世界坐标→体素索引"""
        pos = np.asarray(pos, dtype=np.float64)
        idx = ((pos - self.origin) / self.resolution).astype(int)
        # 转为tuple方便索引
        return (idx[0], idx[1], idx[2])

    def voxel_to_world(self, idx: tuple) -> np.ndarray:
        """体素索引→世界坐标（体素中心）"""
        idx_arr = np.array(idx, dtype=np.float64)
        # 体素中心 = 原点 + (索引 + 0.5) * 分辨率
        return self.origin + (idx_arr + 0.5) * self.resolution

    def _is_valid_voxel(self, idx: tuple) -> bool:
        """检查体素索引是否在有效范围内"""
        return (0 <= idx[0] < self.grid_size[0] and
                0 <= idx[1] < self.grid_size[1] and
                0 <= idx[2] < self.grid_size[2])

    def _raycast_voxels(self, start: np.ndarray, end: np.ndarray) -> list:
        """
        3D DDA射线遍历，获取从start到end经过的所有体素索引

        使用Bresenham 3D直线算法的变体
        返回体素索引列表（不包含end所在的体素）
        """
        start_idx = np.array(self.world_to_voxel(start), dtype=np.float64)
        end_idx = np.array(self.world_to_voxel(end), dtype=np.float64)

        direction = end_idx - start_idx
        n_steps = int(np.max(np.abs(direction))) + 1

        if n_steps <= 1:
            return []

        # 步进方向
        step = np.sign(direction).astype(int)
        step[step == 0] = 1  # 避免零步进

        # 每步的增量
        delta = direction / n_steps

        voxels = []
        current = start_idx.copy()

        for i in range(1, n_steps):  # 跳过起点，不包含终点
            current = start_idx + delta * i
            idx = tuple(current.astype(int))
            if self._is_valid_voxel(idx):
                voxels.append(idx)

        return voxels

    def insert_point(self, point: np.ndarray, free_points: np.ndarray = None):
        """
        插入一个观测点（射线更新）

        逻辑：
        1. 从传感器原点到观测点画射线
        2. 射线经过的体素：概率降低（标记为空闲）
        3. 观测点所在体素：概率提高（标记为占据）

        使用对数概率更新（与真实OctoMap一致）：
        logit(p_new) = logit(p_old) + logit_update
        """
        point = np.asarray(point, dtype=np.float64)

        # 默认传感器原点在地图原点附近（可由free_points指定射线起点）
        if free_points is not None:
            origin = np.asarray(free_points, dtype=np.float64)
        else:
            origin = np.array([0.0, 0.0, 5.0])  # 默认传感器位置

        # 射线遍历：标记空闲体素（排除终点体素）
        end_idx = self.world_to_voxel(point)
        free_voxels = self._raycast_voxels(origin, point)
        for idx in free_voxels:
            # 排除终点体素（终点应标记为占据而非空闲）
            if idx == end_idx:
                continue
            # 空闲更新：logit(p_new) = logit(p_old) + free_update
            p_old = self.grid[idx]
            logit_old = self._logit(float(p_old))
            logit_new = logit_old + self._logit_free_update
            p_new = self._sigmoid(logit_new)
            # 钳位
            self.grid[idx] = np.float32(np.clip(p_new, self.clamp_min, self.clamp_max))

        # 观测点体素：标记为占据
        occ_idx = self.world_to_voxel(point)
        if self._is_valid_voxel(occ_idx):
            p_old = self.grid[occ_idx]
            logit_old = self._logit(float(p_old))
            logit_new = logit_old + self._logit_occ_update
            p_new = self._sigmoid(logit_new)
            self.grid[occ_idx] = np.float32(np.clip(p_new, self.clamp_min, self.clamp_max))

    def insert_pointcloud(self, points: np.ndarray, origin: np.ndarray):
        """批量插入点云"""
        points = np.asarray(points, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)

        if points.ndim == 1:
            points = points.reshape(1, 3)

        for i in range(points.shape[0]):
            self.insert_point(points[i], free_points=origin)

    def get_occupancy(self, pos: np.ndarray) -> float:
        """获取指定位置的占据概率"""
        pos = np.asarray(pos, dtype=np.float64)
        idx = self.world_to_voxel(pos)
        if self._is_valid_voxel(idx):
            return float(self.grid[idx])
        return 0.5  # 地图外返回最大不确定性

    def get_entropy(self, pos: np.ndarray) -> float:
        """获取指定位置的地图熵 H(v) = -[p*log(p) + (1-p)*log(1-p)]"""
        p = self.get_occupancy(pos)
        # 避免log(0)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return -float(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    def get_entropy_sector(self, pos: np.ndarray, n_sectors: int = 8, radius: float = 30.0) -> np.ndarray:
        """
        计算pos周围8个扇区的平均熵值（用于RL观测）

        将周围360度分成n_sectors个扇区，
        每个扇区内体素熵的平均值
        """
        pos = np.asarray(pos, dtype=np.float64)
        sector_entropies = np.zeros(n_sectors, dtype=np.float64)
        sector_counts = np.zeros(n_sectors, dtype=np.int32)

        # 扇区角度范围
        sector_angle = 2.0 * np.pi / n_sectors

        # 在半径范围内采样体素
        # 计算采样范围（体素索引）
        center_idx = np.array(self.world_to_voxel(pos), dtype=np.int32)
        r_voxels = int(np.ceil(radius / self.resolution))

        for di in range(-r_voxels, r_voxels + 1):
            for dj in range(-r_voxels, r_voxels + 1):
                for dk in range(-r_voxels, r_voxels + 1):
                    idx = (center_idx[0] + di, center_idx[1] + dj, center_idx[2] + dk)
                    if not self._is_valid_voxel(idx):
                        continue

                    # 计算体素世界坐标
                    voxel_world = self.voxel_to_world(idx)

                    # 水平距离和角度
                    dx = voxel_world[0] - pos[0]
                    dy = voxel_world[1] - pos[1]
                    horizontal_dist = np.sqrt(dx * dx + dy * dy)
                    dz = abs(voxel_world[2] - pos[2])

                    # 检查是否在半径范围内（3D距离）
                    dist_3d = np.sqrt(horizontal_dist ** 2 + dz ** 2)
                    if dist_3d > radius or horizontal_dist < 1e-6:
                        continue

                    # 计算扇区索引
                    angle = np.arctan2(dy, dx)
                    if angle < 0:
                        angle += 2.0 * np.pi
                    sector_idx = int(angle / sector_angle) % n_sectors

                    # 累加熵
                    p = float(self.grid[idx])
                    p = np.clip(p, 1e-10, 1.0 - 1e-10)
                    h = -float(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
                    sector_entropies[sector_idx] += h
                    sector_counts[sector_idx] += 1

        # 计算平均熵
        for i in range(n_sectors):
            if sector_counts[i] > 0:
                sector_entropies[i] /= sector_counts[i]
            else:
                sector_entropies[i] = 0.0  # 无数据扇区熵为0

        return sector_entropies

    def get_target_direction_entropy(self, pos: np.ndarray, target: np.ndarray) -> float:
        """
        计算从pos到target方向上的平均熵
        （用于RL判断去目标的路上是否安全）
        """
        pos = np.asarray(pos, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return 0.0

        # 沿方向采样点
        n_samples = max(2, int(dist / self.resolution))
        ts = np.linspace(0.0, 1.0, n_samples, endpoint=False)  # 不包含终点

        total_entropy = 0.0
        count = 0
        for t in ts:
            sample_pos = pos + t * direction
            h = self.get_entropy(sample_pos)
            total_entropy += h
            count += 1

        return total_entropy / count if count > 0 else 0.0

    def compute_map_entropy(self) -> float:
        """计算整个地图的总熵"""
        p = self.grid.astype(np.float64)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        return float(np.sum(entropy))

    def reset(self):
        """重置地图"""
        self.grid = np.full(self.grid_size, 0.5, dtype=np.float32)


if __name__ == "__main__":
    print("=== SimpleOctoMap 测试 ===")

    # 使用小地图加速测试
    config = {
        "resolution": 1.0,
        "origin": np.array([-10.0, -10.0, 0.0]),
        "size": np.array([20, 20, 10]),
    }
    omap = SimpleOctoMap(config)
    print(f"网格尺寸: {omap.grid_size}")
    print(f"初始概率: {omap.grid.mean():.4f} (期望0.5)")

    # 测试1: 坐标转换
    world_pos = np.array([0.0, 0.0, 5.0])
    idx = omap.world_to_voxel(world_pos)
    print(f"世界坐标 {world_pos} → 体素索引 {idx}")
    world_back = omap.voxel_to_world(idx)
    print(f"体素索引 {idx} → 世界坐标 {world_back}")
    # 体素中心应接近原始坐标
    assert np.allclose(world_back, world_pos, atol=omap.resolution), "坐标转换不一致"

    # 测试2: 插入观测点
    sensor_origin = np.array([0.0, 0.0, 5.0])
    obs_point = np.array([5.0, 0.0, 5.0])
    print(f"\n插入观测: 传感器={sensor_origin}, 观测点={obs_point}")
    omap.insert_point(obs_point, free_points=sensor_origin)

    # 观测点应被标记为占据
    occ = omap.get_occupancy(obs_point)
    print(f"观测点占据概率: {occ:.4f} (应 > 0.5)")
    assert occ > 0.5, "观测点应被标记为占据"

    # 传感器附近应被标记为空闲
    free_pos = np.array([2.0, 0.0, 5.0])
    free_occ = omap.get_occupancy(free_pos)
    print(f"射线经过点占据概率: {free_occ:.4f} (应 < 0.5)")
    assert free_occ < 0.5, "射线经过点应被标记为空闲"

    # 测试3: 熵计算
    entropy_obs = omap.get_entropy(obs_point)
    entropy_free = omap.get_entropy(free_pos)
    entropy_unknown = omap.get_entropy(np.array([8.0, 8.0, 5.0]))
    print(f"\n观测点熵: {entropy_obs:.4f}")
    print(f"空闲点熵: {entropy_free:.4f}")
    print(f"未知点熵: {entropy_unknown:.4f} (应接近最大熵 ~0.693)")
    # 未知点熵应最大（p=0.5时熵最大）
    assert entropy_unknown > entropy_obs, "未知点熵应大于已观测点"

    # 测试4: 批量点云插入
    omap2 = SimpleOctoMap(config)
    points = np.array([
        [5.0, 0.0, 5.0],
        [0.0, 5.0, 5.0],
        [-5.0, 0.0, 5.0],
    ])
    origin = np.array([0.0, 0.0, 5.0])
    print(f"\n批量插入 {len(points)} 个点")
    omap2.insert_pointcloud(points, origin)
    for p in points:
        occ = omap2.get_occupancy(p)
        print(f"  点 {p} 占据概率: {occ:.4f}")
        assert occ > 0.5, f"点 {p} 应被标记为占据"

    # 测试5: 扇区熵
    sector_entropy = omap2.get_entropy_sector(
        np.array([0.0, 0.0, 5.0]), n_sectors=8, radius=8.0
    )
    print(f"\n8扇区熵: {sector_entropy}")
    assert sector_entropy.shape == (8,), "扇区熵形状不正确"

    # 测试6: 目标方向熵
    target = np.array([5.0, 0.0, 5.0])
    dir_entropy = omap2.get_target_direction_entropy(
        np.array([0.0, 0.0, 5.0]), target
    )
    print(f"到目标方向熵: {dir_entropy:.4f}")
    assert dir_entropy >= 0.0, "方向熵应非负"

    # 测试7: 地图总熵
    total_entropy = omap2.compute_map_entropy()
    print(f"地图总熵: {total_entropy:.2f}")
    assert total_entropy > 0.0, "地图总熵应大于0"

    # 测试8: 重置
    omap2.reset()
    mean_after_reset = omap2.grid.mean()
    print(f"重置后平均概率: {mean_after_reset:.4f} (期望0.5)")
    assert abs(mean_after_reset - 0.5) < 0.01, "重置后概率应回到0.5"

    # 测试9: 对数概率更新一致性
    # 验证多次观测同一占据点，概率应趋近1
    omap3 = SimpleOctoMap(config)
    test_point = np.array([3.0, 0.0, 5.0])
    test_origin = np.array([0.0, 0.0, 5.0])
    probs = []
    for i in range(20):
        omap3.insert_point(test_point, free_points=test_origin)
        p = omap3.get_occupancy(test_point)
        probs.append(p)
    print(f"\n多次观测同一占据点概率变化:")
    for i, p in enumerate(probs[:5]):
        print(f"  第{i+1}次: {p:.4f}")
    print(f"  ... 第20次: {probs[-1]:.4f}")
    # 概率应单调递增
    for i in range(1, len(probs)):
        assert probs[i] >= probs[i-1], "占据概率应单调递增"
    # 最终概率应接近上限
    assert probs[-1] > 0.9, "多次观测后概率应接近1"

    # 测试10: 多次观测空闲，概率趋近0
    omap4 = SimpleOctoMap(config)
    free_test = np.array([2.0, 0.0, 5.0])
    free_probs = []
    for i in range(20):
        omap4.insert_point(np.array([5.0, 0.0, 5.0]), free_points=np.array([0.0, 0.0, 5.0]))
        p = omap4.get_occupancy(free_test)
        free_probs.append(p)
    print(f"\n多次观测后空闲点概率: {free_probs[-1]:.4f} (应趋近0)")
    assert free_probs[-1] < 0.2, "多次观测后空闲点概率应趋近0"

    print("\n=== 所有测试通过 ===")
