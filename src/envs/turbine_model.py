import numpy as np


class TurbineModel:
    """简化圆柱体风机模型"""

    def __init__(self, position: np.ndarray, radius: float = 1.5, height: float = 70.0):
        """
        Args:
            position: 风机底部中心 [x,y,z]
            radius: 塔身半径 m
            height: 塔身高度 m
        """
        self.position = np.asarray(position, dtype=np.float64)
        self.radius = radius
        self.height = height

    def get_surface_points(self, n_theta: int = 36, n_z: int = 20) -> np.ndarray:
        """
        生成塔身表面采样点，用于覆盖评估
        返回 (n_theta * n_z, 3) 数组
        """
        # 角度采样
        thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        # 高度采样
        zs = np.linspace(0.0, self.height, n_z)

        # 生成网格
        theta_grid, z_grid = np.meshgrid(thetas, zs)
        theta_flat = theta_grid.flatten()
        z_flat = z_grid.flatten()

        # 圆柱体表面点
        points = np.zeros((n_theta * n_z, 3))
        points[:, 0] = self.position[0] + self.radius * np.cos(theta_flat)
        points[:, 1] = self.position[1] + self.radius * np.sin(theta_flat)
        points[:, 2] = self.position[2] + z_flat

        return points

    def get_hub_position(self) -> np.ndarray:
        """获取轮毂（塔顶）位置"""
        hub = self.position.copy()
        hub[2] += self.height
        return hub

    def distance_to_surface(self, pos: np.ndarray) -> float:
        """
        计算到塔身表面的最短距离

        逻辑：
        1. 将pos投影到塔身轴心上，得到轴心最近点
        2. 如果投影高度在0~height范围内，距离 = max(0, 到轴心水平距离 - radius)
        3. 如果投影高度超出范围，距离到最近端点
        """
        pos = np.asarray(pos, dtype=np.float64)

        # 水平方向：到轴心的距离
        dx = pos[0] - self.position[0]
        dy = pos[1] - self.position[1]
        horizontal_dist = np.sqrt(dx * dx + dy * dy)

        # 垂直方向：相对于塔底的高度
        dz = pos[2] - self.position[2]

        if 0.0 <= dz <= self.height:
            # 投影在圆柱高度范围内
            # 到表面距离 = 水平距离 - 半径
            return max(0.0, horizontal_dist - self.radius)
        else:
            # 投影超出圆柱高度范围，需要考虑端面和侧面边缘
            if dz < 0.0:
                # 在塔底下方
                dz_bot = -dz
            else:
                # 在塔顶上方
                dz_bot = dz - self.height

            if horizontal_dist <= self.radius:
                # 在端面正上方/正下方，到端面的距离就是dz_bot
                return max(0.0, dz_bot)
            else:
                # 在端面边缘外侧，到端面边缘的距离
                dist_to_rim = np.sqrt((horizontal_dist - self.radius) ** 2 + dz_bot ** 2)
                return max(0.0, dist_to_rim)

    def is_inside(self, pos: np.ndarray) -> bool:
        """检查点是否在圆柱体内"""
        pos = np.asarray(pos, dtype=np.float64)

        # 水平距离
        dx = pos[0] - self.position[0]
        dy = pos[1] - self.position[1]
        horizontal_dist = np.sqrt(dx * dx + dy * dy)

        # 垂直范围
        dz = pos[2] - self.position[2]

        return horizontal_dist <= self.radius and 0.0 <= dz <= self.height


if __name__ == "__main__":
    print("=== TurbineModel 测试 ===")

    # 测试1: 基本创建
    pos = np.array([10.0, 20.0, 0.0])
    turbine = TurbineModel(pos, radius=1.5, height=70.0)
    print(f"风机位置: {pos}, 半径: 1.5m, 高度: 70.0m")

    # 测试2: get_hub_position
    hub = turbine.get_hub_position()
    expected_hub = np.array([10.0, 20.0, 70.0])
    print(f"轮毂位置: {hub}, 期望: {expected_hub}")
    assert np.allclose(hub, expected_hub), "轮毂位置不正确"

    # 测试3: get_surface_points
    surface = turbine.get_surface_points(n_theta=36, n_z=20)
    print(f"表面采样点形状: {surface.shape}")
    assert surface.shape == (36 * 20, 3), "表面点形状不正确"
    # 检查底部点z坐标
    assert abs(surface[0, 2] - 0.0) < 0.01, "底部点z不正确"
    # 检查表面点x坐标范围
    x_range = surface[:, 0].max() - surface[:, 0].min()
    print(f"表面x范围: {x_range:.4f} (期望约 {2 * 1.5:.4f})")
    assert abs(x_range - 2 * 1.5) < 0.01, "表面x范围不正确"

    # 测试4: distance_to_surface - 侧面
    side_point = np.array([15.0, 20.0, 35.0])
    dist = turbine.distance_to_surface(side_point)
    expected_dist = 5.0 - 1.5  # 水平距离5 - 半径1.5
    print(f"侧面点 {side_point} 到表面距离: {dist:.4f}, 期望: {expected_dist:.4f}")
    assert abs(dist - expected_dist) < 0.01, "侧面距离不正确"

    # 测试5: distance_to_surface - 内部
    inside_point = np.array([10.5, 20.0, 35.0])
    dist_inside = turbine.distance_to_surface(inside_point)
    print(f"内部点 {inside_point} 到表面距离: {dist_inside:.4f}, 期望: 0.0")
    assert dist_inside == 0.0, "内部点距离应为0"

    # 测试6: distance_to_surface - 上方
    above_point = np.array([10.0, 20.0, 80.0])
    dist_above = turbine.distance_to_surface(above_point)
    expected_above = 10.0  # 80 - 70 = 10
    print(f"上方点 {above_point} 到表面距离: {dist_above:.4f}, 期望: {expected_above:.4f}")
    assert abs(dist_above - expected_above) < 0.01, "上方距离不正确"

    # 测试7: distance_to_surface - 下方偏移
    below_point = np.array([12.0, 20.0, -3.0])
    dist_below = turbine.distance_to_surface(below_point)
    # 到塔底端面边缘的最短距离
    # horizontal_dist=2 > radius=1.5, 在端面边缘外侧
    # dist_to_rim = sqrt((2-1.5)^2 + 3^2) = sqrt(0.25+9) = sqrt(9.25)
    expected_below = np.sqrt(0.25 + 9.0)
    print(f"下方偏移点 {below_point} 到表面距离: {dist_below:.4f}, 期望: {expected_below:.4f}")
    assert abs(dist_below - expected_below) < 0.01, "下方偏移距离不正确"

    # 测试8: is_inside
    assert turbine.is_inside(np.array([10.0, 20.0, 35.0])) == True, "中心点应在内部"
    assert turbine.is_inside(np.array([11.0, 20.0, 35.0])) == True, "半径内点应在内部"
    assert turbine.is_inside(np.array([12.0, 20.0, 35.0])) == False, "半径外点应不在内部"
    assert turbine.is_inside(np.array([10.0, 20.0, 80.0])) == False, "上方点应不在内部"
    assert turbine.is_inside(np.array([10.0, 20.0, -1.0])) == False, "下方点应不在内部"
    print("is_inside 测试全部通过")

    # 测试9: 表面点数量验证
    surface2 = turbine.get_surface_points(n_theta=12, n_z=5)
    print(f"表面采样点(12x5)形状: {surface2.shape}, 期望: (60, 3)")
    assert surface2.shape == (60, 3), "表面点数量不正确"

    print("\n=== 所有测试通过 ===")
