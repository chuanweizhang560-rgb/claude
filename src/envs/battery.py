import numpy as np


class BatteryModel:
    """无人机电量模型：消耗+基站换电池"""

    def __init__(self, config: dict = None):
        self.capacity = 1.0          # 满电量=1.0
        self.battery = 1.0           # 当前电量
        self.low_threshold = 0.15    # 低电量阈值，低于此规则强制返航

        # 基站参数
        self.base_position = np.array([0.0, 0.0, 0.0])
        self.base_radius = 5.0       # 基站判定半径

        # 功耗系数（与dynamics.py配合）
        self.p_hover = 0.0167        # %/s 悬停功耗（~1%/min, 20min续航）
        self.k_speed = 0.003         # %/(m/s) 速度功耗系数
        self.k_climb = 0.005         # %/(m/s) 爬升额外功耗
        self.k_wind = 0.001          # %/(m/s) 抗风额外功耗
        self.k_accel = 0.002         # %/(m/s²) 大机动额外功耗

        # 从配置字典覆盖默认参数
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    # 如果是列表类型，转为numpy数组
                    if key in ('base_position',) and isinstance(value, (list, tuple)):
                        setattr(self, key, np.array(value))

    def update(self, power: float, dt: float) -> float:
        """
        更新电量

        Args:
            power: 当前功耗 %/s
            dt: 时间步长

        Returns:
            新电量值
        """
        # 消耗电量
        self.battery -= power * dt
        # 电量不能低于0
        self.battery = max(0.0, self.battery)
        # 电量不能超过容量
        self.battery = min(self.capacity, self.battery)
        return self.battery

    def compute_power(self, velocity: np.ndarray, wind: np.ndarray = None,
                      acceleration: np.ndarray = None) -> float:
        """
        计算当前功耗（与dynamics.get_power一致，这里独立实现给env用）

        功耗 = 悬停功耗 + 速度功耗 + 爬升功耗 + 抗风功耗 + 机动功耗

        Args:
            velocity: 速度向量 [vx, vy, vz] m/s
            wind: 风速向量 [wx, wy, wz] m/s（可选）
            acceleration: 加速度向量 [ax, ay, az] m/s²（可选）

        Returns:
            功耗值 %/s
        """
        speed = np.linalg.norm(velocity)
        # 爬升分量：z轴速度为正表示爬升
        climb_speed = max(0.0, velocity[2]) if len(velocity) > 2 else 0.0

        # 基础功耗 = 悬停 + 速度相关
        power = self.p_hover + self.k_speed * speed + self.k_climb * climb_speed

        # 抗风功耗：风速与速度方向相反时额外消耗
        if wind is not None:
            wind_speed = np.linalg.norm(wind)
            power += self.k_wind * wind_speed

        # 大机动额外功耗
        if acceleration is not None:
            accel_mag = np.linalg.norm(acceleration)
            power += self.k_accel * accel_mag

        return power

    def is_low_battery(self) -> bool:
        """是否低电量"""
        return self.battery < self.low_threshold

    def is_dead(self) -> bool:
        """是否电量耗尽"""
        return self.battery <= 0.0

    def try_charge(self, position: np.ndarray) -> bool:
        """
        尝试在基站换电池

        Args:
            position: 无人机当前位置

        Returns:
            是否成功换电（在基站范围内则瞬间满电）
        """
        dist = np.linalg.norm(position - self.base_position)
        if dist <= self.base_radius:
            self.battery = self.capacity
            return True
        return False

    def reset(self):
        """重置电量"""
        self.battery = self.capacity


if __name__ == "__main__":
    # 基本功能测试
    print("=== BatteryModel 测试 ===\n")

    # 1. 初始化测试
    bm = BatteryModel()
    print(f"初始电量: {bm.battery:.2f}")
    assert bm.battery == 1.0, "初始电量应为1.0"

    # 2. 功耗计算测试
    vel = np.array([5.0, 0.0, 1.0])  # 水平5m/s，爬升1m/s
    power = bm.compute_power(vel)
    print(f"速度{vel}的功耗: {power:.4f} %/s")
    expected = bm.p_hover + bm.k_speed * np.linalg.norm(vel) + bm.k_climb * 1.0
    assert abs(power - expected) < 1e-6, f"功耗计算错误: {power} != {expected}"

    # 3. 带风和加速度的功耗
    wind = np.array([3.0, 0.0, 0.0])
    accel = np.array([1.0, 0.0, 0.0])
    power_full = bm.compute_power(vel, wind=wind, acceleration=accel)
    print(f"带风+加速度的功耗: {power_full:.4f} %/s")
    expected_full = (bm.p_hover + bm.k_speed * np.linalg.norm(vel)
                     + bm.k_climb * 1.0 + bm.k_wind * 3.0 + bm.k_accel * 1.0)
    assert abs(power_full - expected_full) < 1e-6, "带风功耗计算错误"

    # 4. 电量更新测试
    bm.update(power=0.02, dt=10.0)  # 消耗0.2
    print(f"消耗后电量: {bm.battery:.2f}")
    assert abs(bm.battery - 0.8) < 1e-6, "电量更新错误"

    # 5. 低电量测试
    bm.battery = 0.1
    print(f"电量0.1时低电量: {bm.is_low_battery()}")
    assert bm.is_low_battery(), "应判定为低电量"

    bm.battery = 0.2
    print(f"电量0.2时低电量: {bm.is_low_battery()}")
    assert not bm.is_low_battery(), "不应判定为低电量"

    # 6. 电量耗尽测试
    bm.battery = 0.0
    print(f"电量0.0时耗尽: {bm.is_dead()}")
    assert bm.is_dead(), "应判定为电量耗尽"

    # 7. 基站换电测试
    bm.battery = 0.3
    pos_near = np.array([3.0, 0.0, 0.0])  # 距基站3m，在半径5m内
    pos_far = np.array([10.0, 0.0, 0.0])  # 距基站10m，在半径外
    print(f"近基站换电: {bm.try_charge(pos_near)}")
    assert bm.try_charge(pos_near), "应在基站范围内换电"
    assert bm.battery == 1.0, "换电后应满电"

    bm.battery = 0.3
    print(f"远基站换电: {bm.try_charge(pos_far)}")
    assert not bm.try_charge(pos_far), "不应在基站范围外换电"
    assert abs(bm.battery - 0.3) < 1e-6, "范围外换电电量不变"

    # 8. 重置测试
    bm.battery = 0.1
    bm.reset()
    print(f"重置后电量: {bm.battery:.2f}")
    assert bm.battery == 1.0, "重置后应满电"

    # 9. 电量下限测试
    bm.battery = 0.5
    bm.update(power=1.0, dt=1.0)  # 消耗1.0，应被截断为0
    print(f"过度消耗后电量: {bm.battery:.2f}")
    assert bm.battery == 0.0, "电量不应低于0"

    # 10. 配置覆盖测试
    bm2 = BatteryModel(config={'capacity': 2.0, 'low_threshold': 0.2})
    print(f"\n配置覆盖后容量: {bm2.capacity}, 低电量阈值: {bm2.low_threshold}")
    assert bm2.capacity == 2.0, "配置覆盖容量失败"
    assert bm2.low_threshold == 0.2, "配置覆盖阈值失败"

    print("\n=== 所有测试通过 ===")
