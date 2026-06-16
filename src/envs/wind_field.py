import numpy as np


class WindField:
    """沙漠风场模型：稳态风 + 湍流（Ornstein-Uhlenbeck过程）+ 可选阵风"""

    def __init__(self, config: dict = None):
        # 稳态风参数
        self.steady_speed = 3.0          # m/s 稳态风速
        self.steady_direction = 0.0      # rad 稳态风方向（从x轴逆时针）

        # 湍流参数
        self.turbulence_intensity = 0.3  # 湍流强度（相对稳态风的比例）
        self.turbulence_freq = 1.0       # Hz 湍流频率（OU过程的均值回复速率）

        # 阵风参数（后期课程加入，默认关闭）
        self.gust_enabled = False
        self.gust_speed = 10.0           # m/s 阵风风速
        self.gust_duration = 3.0         # s 阵风持续时间
        self.gust_probability = 0.01     # 每步触发概率

        # 内部状态
        self.rng = np.random.default_rng(42)
        self.time = 0.0
        self.dt = 0.05

        # OU过程当前状态 [ou_x, ou_y, ou_z]
        self._ou_state = np.zeros(3)

        # 阵风状态
        self._gust_active = False        # 当前是否有阵风
        self._gust_remaining = 0.0       # 阵风剩余时间（秒）
        self._gust_direction = 0.0       # 阵风方向（弧度）
        self._gust_elevation = 0.0       # 阵风仰角（弧度）

        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def _steady_wind(self) -> np.ndarray:
        """计算稳态风向量 [wx, wy, wz]

        稳态风为水平方向，wz=0。
        """
        wx = self.steady_speed * np.cos(self.steady_direction)
        wy = self.steady_speed * np.sin(self.steady_direction)
        wz = 0.0
        return np.array([wx, wy, wz])

    def _turbulence_wind(self) -> np.ndarray:
        """用Ornstein-Uhlenbeck过程生成时相关的湍流风

        OU过程的离散形式:
            x(t+dt) = x(t) + theta * (mu - x(t)) * dt + sigma * sqrt(dt) * N(0,1)

        其中:
            theta = 2 * pi * turbulence_freq  (均值回复速率)
            mu = 0                             (长期均值为0)
            sigma = steady_speed * turbulence_intensity  (波动幅度)
        """
        theta = 2 * np.pi * self.turbulence_freq  # 均值回复速率
        sigma = self.steady_speed * self.turbulence_intensity  # 波动标准差
        sqrt_dt = np.sqrt(self.dt)

        # 三个分量独立OU过程
        noise = self.rng.standard_normal(3)
        self._ou_state = (self._ou_state
                          + theta * (0.0 - self._ou_state) * self.dt
                          + sigma * sqrt_dt * noise)

        # 湍流主要为水平分量，垂直分量衰减
        turbulence = self._ou_state.copy()
        turbulence[2] *= 0.3  # 垂直湍流较弱

        return turbulence

    def _gust_wind(self) -> np.ndarray:
        """计算阵风向量

        每步以 gust_probability 概率触发阵风。
        触发后持续 gust_duration 秒，方向随机，强度为 gust_speed。
        """
        if not self.gust_enabled:
            return np.zeros(3)

        # 如果没有活跃阵风，尝试触发
        if not self._gust_active:
            if self.rng.random() < self.gust_probability:
                self._gust_active = True
                self._gust_remaining = self.gust_duration
                # 阵风方向随机（水平0~2pi，仰角-10~10度）
                self._gust_direction = self.rng.uniform(0, 2 * np.pi)
                self._gust_elevation = self.rng.uniform(np.radians(-10), np.radians(10))

        # 如果有活跃阵风，计算阵风向量
        if self._gust_active:
            self._gust_remaining -= self.dt
            if self._gust_remaining <= 0.0:
                # 阵风结束
                self._gust_active = False
                return np.zeros(3)

            # 阵风强度随时间线性衰减（更自然的效果）
            ratio = self._gust_remaining / self.gust_duration
            speed = self.gust_speed * ratio

            wx = speed * np.cos(self._gust_direction) * np.cos(self._gust_elevation)
            wy = speed * np.sin(self._gust_direction) * np.cos(self._gust_elevation)
            wz = speed * np.sin(self._gust_elevation)
            return np.array([wx, wy, wz])

        return np.zeros(3)

    def get_wind(self, position: np.ndarray = None) -> np.ndarray:
        """获取当前位置的风速向量 [wx, wy, wz]

        风速 = 稳态风 + 湍流 + 可选阵风

        参数:
            position: 位置向量 [x, y, z]，当前版本不使用空间变化，预留接口
        返回:
            风速向量 [wx, wy, wz] (m/s)
        """
        wind = (self._steady_wind()
                + self._turbulence_wind()
                + self._gust_wind())

        # 推进时间
        self.time += self.dt

        return wind

    def reset(self, seed: int = None):
        """重置风场，每个episode调用

        参数:
            seed: 随机种子，None则不重置随机数生成器
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.rng.integers(0, 2**31))

        self.time = 0.0
        self._ou_state = np.zeros(3)
        self._gust_active = False
        self._gust_remaining = 0.0


if __name__ == "__main__":
    # 基本功能测试
    wind = WindField()

    # 1. 稳态风测试
    steady = wind._steady_wind()
    print(f"稳态风: [{steady[0]:.4f}, {steady[1]:.4f}, {steady[2]:.4f}] m/s")
    expected_wx = 3.0 * np.cos(0.0)
    expected_wy = 3.0 * np.sin(0.0)
    assert abs(steady[0] - expected_wx) < 1e-9, f"稳态风wx错误: {steady[0]}"
    assert abs(steady[1] - expected_wy) < 1e-9, f"稳态风wy错误: {steady[1]}"
    assert abs(steady[2]) < 1e-9, f"稳态风wz应为0: {steady[2]}"
    print("稳态风测试通过!")

    # 2. 连续采样风场，检查湍流时相关性
    wind.reset(seed=42)
    samples = []
    for _ in range(100):
        w = wind.get_wind()
        samples.append(w)
    samples = np.array(samples)
    print(f"\n100步采样 - 平均风速: [{samples[:, 0].mean():.4f}, {samples[:, 1].mean():.4f}, {samples[:, 2].mean():.4f}] m/s")
    print(f"100步采样 - 风速标准差: [{samples[:, 0].std():.4f}, {samples[:, 1].std():.4f}, {samples[:, 2].std():.4f}] m/s")
    # 湍流应该让风速围绕稳态值波动，但不会偏离太远
    # x方向应该接近稳态风速3.0
    print(f"x方向平均风速接近稳态值{wind.steady_speed}: {abs(samples[:, 0].mean() - wind.steady_speed) < 1.0}")

    # 3. 阵风测试
    wind_gust = WindField(config={
        "gust_enabled": True,
        "gust_speed": 10.0,
        "gust_duration": 3.0,
        "gust_probability": 1.0,  # 必定触发，方便测试
    })
    wind_gust.reset(seed=42)
    # 第一步必然触发阵风
    w0 = wind_gust.get_wind()
    wind_speed = np.linalg.norm(w0)
    print(f"\n阵风开启时第1步风速: {wind_speed:.4f} m/s (期望 > 稳态风 {wind_gust.steady_speed})")
    assert wind_speed > wind_gust.steady_speed, f"阵风未生效: {wind_speed}"

    # 阵风应在gust_duration后结束
    # 持续采样直到阵风结束
    gust_found = False
    gust_ended = False
    for i in range(200):
        w = wind_gust.get_wind()
        ws = np.linalg.norm(w)
        if ws > wind_gust.steady_speed + 2.0:
            gust_found = True
        if gust_found and ws < wind_gust.steady_speed + 2.0:
            gust_ended = True
            break
    print(f"阵风触发: {gust_found}, 阵风结束: {gust_ended}")

    # 4. 重置测试
    wind.reset(seed=100)
    assert wind.time == 0.0, f"重置后时间应为0: {wind.time}"
    assert np.allclose(wind._ou_state, 0.0), f"重置后OU状态应为0: {wind._ou_state}"
    print("\n重置测试通过!")

    # 5. 自定义方向测试
    wind_diag = WindField(config={"steady_direction": np.pi / 4})  # 45度
    steady_diag = wind_diag._steady_wind()
    print(f"\n45度稳态风: [{steady_diag[0]:.4f}, {steady_diag[1]:.4f}] m/s")
    assert abs(steady_diag[0] - steady_diag[1]) < 1e-9, "45度方向wx应等于wy"
    print("方向测试通过!")

    # 6. 长时间统计测试（无阵风）
    wind_stat = WindField(config={"turbulence_intensity": 0.3})
    wind_stat.reset(seed=42)
    long_samples = []
    for _ in range(2000):
        long_samples.append(wind_stat.get_wind())
    long_samples = np.array(long_samples)
    print(f"\n2000步统计 - x方向均值: {long_samples[:, 0].mean():.4f} (期望接近 {wind_stat.steady_speed})")
    print(f"2000步统计 - x方向标准差: {long_samples[:, 0].std():.4f}")
    print(f"2000步统计 - z方向标准差: {long_samples[:, 2].std():.4f} (期望较小)")

    print("\n所有测试通过!")
