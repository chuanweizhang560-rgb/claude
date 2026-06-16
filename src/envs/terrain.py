import numpy as np


class DesertTerrain:
    """程序化沙漠地形heightmap，与Gazebo中gen_desert_heightmap.py逻辑一致"""

    def __init__(self, config: dict = None):
        # 默认参数
        self.size = 300.0          # 地形尺寸 300m x 300m
        self.max_height = 8.0      # 最大沙丘高度 8m
        self.resolution = 257      # heightmap分辨率 257x257
        self.seed = 42

        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # 生成后缓存的heightmap
        self._heightmap = None
        # 每个网格点对应的实际坐标范围
        self._x_coords = None
        self._y_coords = None

    def generate(self) -> np.ndarray:
        """生成heightmap，返回 (resolution, resolution) 的numpy数组，值范围[0, max_height]

        多频正弦波组合模拟风成沙丘：
        - 频率1: 大沙丘，低频，振幅4m，波长100m，方向沿x轴（主风向）
        - 频率2: 中等沙丘，振幅2m，波长50m，方向偏30度
        - 频率3: 小波纹，高频，振幅0.5m，波长15m，方向随机
        """
        rng = np.random.default_rng(self.seed)

        # 创建归一化网格坐标 [0, 1]
        lin = np.linspace(0, 1, self.resolution)
        gx, gy = np.meshgrid(lin, lin)

        # 实际物理坐标（米）
        px = gx * self.size  # 0 ~ 300m
        py = gy * self.size  # 0 ~ 300m

        heightmap = np.zeros((self.resolution, self.resolution), dtype=np.float64)

        # ---- 频率1: 大沙丘 ----
        # 振幅4m，波长100m，方向沿x轴（主风向，0度）
        amp1 = 4.0
        wl1 = 100.0           # 波长（米）
        dir1 = 0.0            # 方向角（弧度）
        k1 = 2 * np.pi / wl1  # 波数
        # 沿方向投影
        proj1 = px * np.cos(dir1) + py * np.sin(dir1)
        heightmap += amp1 * np.sin(k1 * proj1)

        # ---- 频率2: 中等沙丘 ----
        # 振幅2m，波长50m，方向偏30度
        amp2 = 2.0
        wl2 = 50.0
        dir2 = np.radians(30.0)
        k2 = 2 * np.pi / wl2
        proj2 = px * np.cos(dir2) + py * np.sin(dir2)
        heightmap += amp2 * np.sin(k2 * proj2)

        # ---- 频率3: 小波纹 ----
        # 振幅0.5m，波长15m，方向随机
        amp3 = 0.5
        wl3 = 15.0
        dir3 = rng.uniform(0, 2 * np.pi)
        k3 = 2 * np.pi / wl3
        proj3 = px * np.cos(dir3) + py * np.sin(dir3)
        heightmap += amp3 * np.sin(k3 * proj3)

        # ---- 归一化到 [0, max_height] ----
        h_min = heightmap.min()
        h_max = heightmap.max()
        if h_max - h_min > 1e-9:
            heightmap = (heightmap - h_min) / (h_max - h_min) * self.max_height
        else:
            # 退化情况：所有高度相同
            heightmap = np.full_like(heightmap, self.max_height / 2)

        # 缓存结果
        self._heightmap = heightmap
        self._x_coords = np.linspace(0, self.size, self.resolution)
        self._y_coords = np.linspace(0, self.size, self.resolution)

        return heightmap

    def get_height(self, x: float, y: float) -> float:
        """查询指定(x,y)位置的地形高度，双线性插值

        坐标范围: x ∈ [0, size], y ∈ [0, size]
        超出范围的位置会被钳位到边界。
        """
        if self._heightmap is None:
            self.generate()

        # 钳位到有效范围
        x = np.clip(x, 0.0, self.size)
        y = np.clip(y, 0.0, self.size)

        # 将物理坐标转换为网格索引（浮点）
        dx = self.size / (self.resolution - 1)
        fi = x / dx  # 列索引
        fj = y / dx  # 行索引

        # 取整数部分和小数部分
        i0 = int(np.floor(fi))
        j0 = int(np.floor(fj))
        i1 = min(i0 + 1, self.resolution - 1)
        j1 = min(j0 + 1, self.resolution - 1)

        # 防止越界
        i0 = max(0, min(i0, self.resolution - 1))
        j0 = max(0, min(j0, self.resolution - 1))

        di = fi - i0
        dj = fj - j0

        # 双线性插值
        h00 = self._heightmap[j0, i0]
        h10 = self._heightmap[j0, i1]
        h01 = self._heightmap[j1, i0]
        h11 = self._heightmap[j1, i1]

        h = (h00 * (1 - di) * (1 - dj)
             + h10 * di * (1 - dj)
             + h01 * (1 - di) * dj
             + h11 * di * dj)

        return float(h)

    def is_collision(self, pos: np.ndarray, safe_margin: float = 1.0) -> bool:
        """检查位置是否与地形碰撞：pos.z < terrain_height + safe_margin

        参数:
            pos: 位置向量 [x, y, z]，z为高度
            safe_margin: 安全余量（米），碰撞判定在 terrain_height + safe_margin 以下
        返回:
            True 表示碰撞
        """
        terrain_h = self.get_height(pos[0], pos[1])
        return pos[2] < terrain_h + safe_margin


if __name__ == "__main__":
    # 基本功能测试
    terrain = DesertTerrain()

    # 1. 生成heightmap并检查形状和值范围
    hmap = terrain.generate()
    print(f"heightmap形状: {hmap.shape}")
    print(f"高度范围: [{hmap.min():.4f}, {hmap.max():.4f}] m")
    assert hmap.shape == (257, 257), f"形状错误: {hmap.shape}"
    assert hmap.min() >= 0.0, f"最小值为负: {hmap.min()}"
    assert hmap.max() <= 8.0 + 1e-6, f"最大值超出: {hmap.max()}"

    # 2. 查询指定点高度
    h_center = terrain.get_height(150.0, 150.0)
    print(f"中心点(150, 150)高度: {h_center:.4f} m")

    h_corner = terrain.get_height(0.0, 0.0)
    print(f"角落(0, 0)高度: {h_corner:.4f} m")

    # 3. 边界外查询（应钳位）
    h_out = terrain.get_height(-10.0, 500.0)
    print(f"边界外(-10, 500)高度: {h_out:.4f} m (应与边界值相同)")

    # 4. 碰撞检测
    # 在地形上方 -> 不碰撞
    pos_safe = np.array([150.0, 150.0, h_center + 5.0])
    print(f"安全位置碰撞: {terrain.is_collision(pos_safe)} (期望 False)")

    # 在地形下方 -> 碰撞
    pos_collide = np.array([150.0, 150.0, h_center - 1.0])
    print(f"碰撞位置碰撞: {terrain.is_collision(pos_collide)} (期望 True)")

    # 在地形附近，考虑安全余量
    pos_margin = np.array([150.0, 150.0, h_center + 0.5])
    print(f"余量内位置碰撞(余量1m): {terrain.is_collision(pos_margin, safe_margin=1.0)} (期望 True)")
    print(f"余量外位置碰撞(余量0.1m): {terrain.is_collision(pos_margin, safe_margin=0.1)} (期望 False)")

    # 5. 自定义配置测试
    terrain_custom = DesertTerrain(config={"size": 100.0, "max_height": 4.0, "seed": 123})
    hmap_custom = terrain_custom.generate()
    print(f"\n自定义地形 - 形状: {hmap_custom.shape}, 高度范围: [{hmap_custom.min():.4f}, {hmap_custom.max():.4f}] m")

    print("\n所有测试通过!")
