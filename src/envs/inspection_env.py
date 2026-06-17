"""电力巡检Gymnasium环境：残差RL架构

规则控制器输出引导速度，RL输出修正量，叠加后送入动力学模型。
观测24维，动作5维 [Δvx, Δvy, Δvz, Δyaw_rate, task_select]。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 确保项目根目录在sys.path中
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.envs.dynamics import QuadrotorDynamics
from src.envs.terrain import DesertTerrain
from src.envs.wind_field import WindField
from src.envs.battery import BatteryModel
from src.envs.coverage import CoverageEvaluator
from src.envs.collision import CollisionChecker
from src.envs.cable_model import CableModel
from src.envs.turbine_model import TurbineModel
from src.envs.octomap_simple import SimpleOctoMap
from src.controllers.turbine_orbit import TurbineOrbitController
from src.controllers.cable_follow import CableFollowController
from src.controllers.explore_guide import ExploreGuideController
from src.controllers.return_home import ReturnHomeController
from src.configs.world_config import get_phase1_config


class InspectionEnv(gym.Env):
    """电力巡检多无人机协同环境（单机版本）

    残差RL架构：
        vel_cmd = rule_vel + Δaction
        rule_vel 由规则控制器根据当前任务生成
        Δaction 由RL策略输出
    """

    metadata = {"render_modes": ["human"]}

    # 任务类型映射
    TASK_TURBINE = 0
    TASK_CABLE = 1
    TASK_EXPLORE = 2
    TASK_RETURN = 3

    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__()

        # 加载配置
        if config is None:
            config = get_phase1_config()
        self.config = config

        self.render_mode = render_mode

        # === 场景参数 ===
        self.world_size = config.get("world_size", 300.0)
        self.max_steps = config.get("max_steps", 2000)
        self.randomize_start = config.get("randomize_start", True)
        self.start_pos_range = config.get("start_pos_range", 5.0)
        self.randomize_wind = config.get("randomize_wind", True)

        # === 动作缩放 ===
        action_scale = config.get("action_scale", {})
        self.scale_h = action_scale.get("horizontal", 1.0)
        self.scale_v = action_scale.get("vertical", 0.5)
        self.scale_yaw = action_scale.get("yaw_rate", 0.5)

        # === 奖励权重 ===
        rw = config.get("reward_weights", {})
        self.w_coverage = rw.get("coverage", 10.0)
        self.w_collision = rw.get("collision", 5.0)
        self.w_power = rw.get("power", 0.1)
        self.w_battery_dead = rw.get("battery_dead", 20.0)
        self.w_coverage_bonus = rw.get("coverage_bonus", 50.0)

        # === 初始化模块 ===
        self._init_modules(config)

        # === Gymnasium 空间 ===
        # 观测空间 24维，所有值归一化到大致[-1,1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(24,), dtype=np.float32
        )

        # 动作空间 5维: [Δvx, Δvy, Δvz, Δyaw_rate, task_select]
        # 前4维连续 ∈ [-1,1]，第5维离散 ∈ {0,1,2,3}
        # 使用Box统一处理，task_select用argmax
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 内部状态
        self.step_count = 0
        self.total_reward = 0.0
        self.collision_count = 0
        self.prev_coverage = 0.0
        self.current_task = self.TASK_TURBINE  # 初始任务：风机巡检

        # 缓存（减少OctoMap计算频率）
        self._cached_entropy_sectors = np.zeros(8)
        self._cached_dir_entropy = 0.0
        self._entropy_update_interval = 50  # 每50步更新一次扇区熵

    def _init_modules(self, config: dict):
        """初始化所有环境模块"""
        # 动力学
        self.dynamics = QuadrotorDynamics(config.get("dynamics"))
        self.dt = self.dynamics.dt

        # 地形
        self.terrain = DesertTerrain(config.get("terrain"))
        self.terrain.generate()

        # 风场
        self.wind_field = WindField(config.get("wind"))

        # 电池
        battery_config = config.get("battery", {})
        battery_config["base_position"] = config.get("base_position", np.array([0.0, 0.0, 0.0]))
        battery_config["base_radius"] = config.get("base_radius", 5.0)
        self.battery = BatteryModel(battery_config)

        # 覆盖评估
        self.coverage_eval = CoverageEvaluator(config.get("coverage"))
        self.coverage_threshold = self.coverage_eval.coverage_dist_turbine  # 用于判断覆盖完成

        # 碰撞检测
        self.collision = CollisionChecker(config.get("collision"))

        # OctoMap — 训练用较小地图加速
        octomap_cfg = config.get("octomap", {}).copy()
        # 训练加速：用更粗分辨率和小范围
        octomap_cfg.setdefault("resolution", 2.0)
        octomap_cfg.setdefault("origin", np.array([-75.0, -75.0, 0.0]))
        octomap_cfg.setdefault("size", np.array([150, 150, 50]))
        self.octomap = SimpleOctoMap(octomap_cfg)

        # 风机和电缆
        self.turbines = []
        for t_cfg in config.get("turbines", []):
            self.turbines.append(TurbineModel(
                position=t_cfg["position"],
                radius=t_cfg.get("radius", 1.5),
                height=t_cfg.get("height", 70.0),
            ))

        self.cables = []
        for c_cfg in config.get("cables", []):
            self.cables.append(CableModel(
                start=c_cfg["start"],
                end=c_cfg["end"],
                sag=c_cfg.get("sag", 4.0),
            ))

        # 预计算电缆采样点
        self.cable_points_list = []
        for cable in self.cables:
            n_pts = 50
            for c_cfg in config.get("cables", []):
                n_pts = c_cfg.get("n_points", 50)
            self.cable_points_list.append(cable.get_points(n_pts))

        # 基站
        self.base_position = np.array(config.get("base_position", [0.0, 0.0, 0.0]),
                                       dtype=np.float64)
        self.base_radius = config.get("base_radius", 5.0)

        # 规则控制器
        self.controllers = {
            self.TASK_TURBINE: TurbineOrbitController(),
            self.TASK_CABLE: CableFollowController(),
            self.TASK_EXPLORE: ExploreGuideController(),
            self.TASK_RETURN: ReturnHomeController(),
        }

        # UAV参数
        self.uav_start_pos = np.array(config.get("uav_start_pos", [5.0, 5.0, 5.0]),
                                       dtype=np.float64)
        self.uav_role = config.get("uav_role", "primary")
        self.uav_group_id = config.get("uav_group_id", 0)

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置各模块
        self.wind_field.reset(seed=seed)
        self.battery.reset()
        self.octomap.reset()
        for ctrl in self.controllers.values():
            if hasattr(ctrl, 'reset'):
                ctrl.reset()

        # UAV状态
        start_pos = self.uav_start_pos.copy()
        if self.randomize_start and seed is not None:
            rng = np.random.default_rng(seed)
            start_pos[:2] += rng.uniform(-self.start_pos_range, self.start_pos_range, 2)
            start_pos[2] = max(3.0, start_pos[2] + rng.uniform(-2.0, 2.0))

        self.uav_state = {
            'pos': start_pos.tolist(),
            'vel': [0.0, 0.0, 0.0],
            'yaw': 0.0,
        }

        # 随机风向偏移
        if self.randomize_wind and seed is not None:
            rng = np.random.default_rng(seed + 1)
            self.wind_field.steady_direction = rng.uniform(0, 2 * np.pi)

        # 重置覆盖状态
        self.turbine_coverage = [0.0] * len(self.turbines)
        self.cable_coverage = [0.0] * len(self.cables)
        self.prev_coverage = 0.0
        self.overall_coverage = 0.0

        # 重置计数
        self.step_count = 0
        self.total_reward = 0.0
        self.collision_count = 0
        self.current_task = self.TASK_TURBINE  # 从风机开始

        # 获取初始观测
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """执行一步

        Args:
            action: [Δvx, Δvy, Δvz, Δyaw_rate, task_select]
                    前4维 ∈ [-1,1]连续，第5维用argmax选任务

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32)

        # === 1. 解析动作 ===
        delta_vel = np.array([
            action[0] * self.scale_h,       # Δvx
            action[1] * self.scale_h,       # Δvy
            action[2] * self.scale_v,       # Δvz
            action[3] * self.scale_yaw,     # Δyaw_rate
        ])

        # 任务选择（阶段1: 简化逻辑）
        # action[4] > 0 → 切到下一个任务; 低电量强制返航
        if self.battery.is_low_battery():
            self.current_task = self.TASK_RETURN
        elif action[4] > 0.0:
            # 按顺序切换: turbine → cable → explore → return
            self.current_task = min(self.current_task + 1, self.TASK_RETURN)

        # === 2. 规则控制器引导速度 ===
        rule_state = self._build_controller_state()
        controller = self.controllers[self.current_task]
        rule_vel = controller.get_guided_velocity(rule_state)

        # === 3. 残差叠加 ===
        vel_cmd = np.zeros(4)
        vel_cmd[:3] = rule_vel[:3] + delta_vel[:3]
        vel_cmd[3] = rule_vel[3] + delta_vel[3]

        # === 4. 碰撞裁剪 ===
        obstacles = self._get_obstacles()
        vel_3d = self.collision.clip_action(
            vel_cmd[:3],
            np.array(self.uav_state['pos']),
            obstacles,
            dt=self.dt,
        )
        vel_cmd[:3] = vel_3d

        # === 5. 动力学更新 ===
        wind = self.wind_field.get_wind(np.array(self.uav_state['pos']))
        self.uav_state = self.dynamics.step(self.uav_state, vel_cmd, wind)

        # 地面约束：不低于地形高度+安全裕度
        terrain_h = self.terrain.get_height(self.uav_state['pos'][0], self.uav_state['pos'][1])
        min_alt = terrain_h + self.collision.ground_margin
        if self.uav_state['pos'][2] < min_alt:
            self.uav_state['pos'][2] = min_alt
            self.uav_state['vel'][2] = max(0.0, self.uav_state['vel'][2])

        # === 6. 电池更新 ===
        power = self.battery.compute_power(
            np.array(self.uav_state['vel']),
            wind=wind,
        )
        self.battery.update(power, self.dt)

        # 基站换电
        self.battery.try_charge(np.array(self.uav_state['pos']))

        # === 7. OctoMap更新（阶段1跳过，节省计算） ===
        # if self.step_count % 20 == 0:
        #     self._update_octomap()

        # === 8. 覆盖评估 ===
        self._update_coverage()
        delta_coverage = self.overall_coverage - self.prev_coverage
        self.prev_coverage = self.overall_coverage

        # === 9. 碰撞检测（用于奖励） ===
        is_collision, _ = self.collision.check_all(
            np.array(self.uav_state['pos']),
            obstacles,
        )
        if is_collision:
            self.collision_count += 1

        # === 10. 计算奖励 ===
        reward = self._compute_reward(delta_coverage, is_collision, power)

        # === 11. 终止条件 ===
        terminated = self._check_coverage_done()
        truncated = False

        # 全覆盖完成额外奖励
        if terminated:
            reward += self.w_coverage_bonus

        self.step_count += 1
        self.total_reward += reward

        # 超时或电量耗尽
        if self.step_count >= self.max_steps:
            truncated = True
        if self.battery.is_dead():
            truncated = True
            reward -= self.w_battery_dead  # 电量耗尽惩罚

        obs = self._get_obs()
        info = self._get_info()
        info["delta_coverage"] = delta_coverage
        info["is_collision"] = is_collision

        return obs, float(reward), terminated, truncated, info

    def _build_controller_state(self) -> dict:
        """构建规则控制器的state输入"""
        # 根据当前任务构建target
        if self.current_task == self.TASK_TURBINE and len(self.turbines) > 0:
            target = {
                'type': 'turbine',
                'position': self.turbines[0].position,
                'radius': self.turbines[0].radius,
                'height': self.turbines[0].height,
            }
        elif self.current_task == self.TASK_CABLE and len(self.cables) > 0:
            target = {
                'type': 'cable',
                'points': self.cable_points_list[0],
                'start': self.cables[0].start,
                'end': self.cables[0].end,
            }
        else:
            target = {}

        state = {
            'pos': self.uav_state['pos'],
            'vel': self.uav_state['vel'],
            'yaw': self.uav_state['yaw'],
            'target': target,
            'role': self.uav_role,
            'battery': self.battery.battery,
            'base_pos': self.base_position.tolist(),
        }

        # ExploreGuide需要entropy_map
        if self.current_task == self.TASK_EXPLORE:
            pos = np.array(self.uav_state['pos'])
            state['entropy_map'] = self.octomap.get_entropy_sector(pos, n_sectors=8, radius=30.0)
        else:
            state['entropy_map'] = np.ones(8) * 0.5

        return state

    def _get_obstacles(self) -> dict:
        """获取当前障碍物字典"""
        obstacles = {
            'terrain_height': self.terrain.get_height(
                self.uav_state['pos'][0], self.uav_state['pos'][1]
            ),
            'turbines': [(t.position, t.radius) for t in self.turbines],
            'cables': self.cable_points_list,
            'other_uavs': [],  # 单机模式无其他UAV
        }
        return obstacles

    def _update_octomap(self):
        """更新OctoMap（简化版：仅标记障碍物附近为占据，UAV附近为空闲）"""
        pos = np.array(self.uav_state['pos'])

        # 标记UAV附近体素为空闲
        for dx in [-5, 0, 5]:
            for dy in [-5, 0, 5]:
                free_pos = pos + np.array([dx, dy, 0.0])
                idx = self.octomap.world_to_voxel(free_pos)
                if self.octomap._is_valid_voxel(idx):
                    p_old = float(self.octomap.grid[idx])
                    logit_old = self.octomap._logit(p_old)
                    logit_new = logit_old + self.octomap._logit_free_update
                    p_new = self.octomap._sigmoid(logit_new)
                    self.octomap.grid[idx] = np.float32(
                        np.clip(p_new, self.octomap.clamp_min, self.octomap.clamp_max)
                    )

        # 标记可见障碍物为占据
        for turbine in self.turbines:
            dist = turbine.distance_to_surface(pos)
            if dist < 50.0:  # 传感器范围
                # 标记风机表面几个点
                for h_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    occ_pos = turbine.position.copy()
                    occ_pos[2] += turbine.height * h_frac
                    self.octomap.insert_point(occ_pos, free_points=pos)

        for cable_pts in self.cable_points_list:
            # 标记电缆中最近几个点
            diffs = cable_pts - pos
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] < 50.0:
                for offset in [-2, -1, 0, 1, 2]:
                    idx = nearest_idx + offset
                    if 0 <= idx < len(cable_pts):
                        self.octomap.insert_point(cable_pts[idx], free_points=pos)

    def _update_coverage(self):
        """更新覆盖进度"""
        pos = np.array(self.uav_state['pos'])

        # 风机覆盖
        for i, turbine in enumerate(self.turbines):
            cov = self.coverage_eval.compute_turbine_coverage(pos, turbine.position)
            self.turbine_coverage[i] = max(self.turbine_coverage[i], cov)

        # 电缆覆盖
        for i, cable_pts in enumerate(self.cable_points_list):
            cov = self.coverage_eval.compute_cable_coverage(pos, cable_pts)
            self.cable_coverage[i] = max(self.cable_coverage[i], cov)

        # 总覆盖率（所有目标的平均）
        all_coverage = self.turbine_coverage + self.cable_coverage
        self.overall_coverage = np.mean(all_coverage) if all_coverage else 0.0

    def _check_coverage_done(self) -> bool:
        """检查是否所有目标都已覆盖"""
        for cov in self.turbine_coverage:
            if not self.coverage_eval.check_turbine_covered(cov):
                return False
        for cov in self.cable_coverage:
            if not self.coverage_eval.check_cable_covered(cov):
                return False
        return True

    def _compute_reward(self, delta_coverage: float, is_collision: bool, power: float) -> float:
        """计算奖励"""
        reward = 0.0

        # 覆盖奖励（主要驱动）
        reward += self.w_coverage * delta_coverage

        # 碰撞惩罚
        if is_collision:
            reward -= self.w_collision

        # 功耗惩罚（轻微）
        reward -= self.w_power * power * self.dt

        return reward

    def _get_obs(self) -> np.ndarray:
        """构建24维观测向量，归一化到大致[-1,1]"""
        pos = np.array(self.uav_state['pos'])
        vel = np.array(self.uav_state['vel'])
        yaw = self.uav_state['yaw']

        obs = np.zeros(24, dtype=np.float32)

        # 1. 自身状态(7): pos_xyz(归一化), vel_xyz, battery
        obs[0] = pos[0] / self.world_size * 2 - 1      # x ∈ [-1,1]
        obs[1] = pos[1] / self.world_size * 2 - 1      # y ∈ [-1,1]
        obs[2] = (pos[2] - 35.0) / 35.0                 # z: 0~70 → [-1,1]
        obs[3] = vel[0] / self.dynamics.max_vel          # vx / max_vel
        obs[4] = vel[1] / self.dynamics.max_vel          # vy / max_vel
        obs[5] = vel[2] / self.dynamics.max_vel_z        # vz / max_vel_z
        obs[6] = self.battery.battery                    # [0,1]

        # 2. 姿态(1): yaw
        obs[7] = yaw / np.pi                              # [-1,1]

        # 3. 任务状态(4)
        obs[8] = float(self.current_task) / 3.0          # task_type ∈ [0,1]
        # 目标距离
        target_dist = self._get_target_distance()
        obs[9] = np.clip(target_dist / 150.0, -1, 1)     # 归一化
        # 覆盖进度
        obs[10] = self.overall_coverage * 2 - 1           # [0,1] → [-1,1]
        # task_select_mask: 阶段1固定为1（不能自由选任务）
        obs[11] = 1.0

        # 4. 角色(2)
        obs[12] = 1.0 if self.uav_role == "primary" else 0.0
        obs[13] = float(self.uav_group_id)               # 0 or 1

        # 5. 地图信息(9): 8扇区熵 + 目标方向熵
        # 阶段1简化：用距离信息代替熵（覆盖任务不需要地图熵驱动）
        # 扇区熵用目标方向编码替代：8扇区中指向当前目标方向的扇区=1，其余衰减
        target_pos = self._get_current_target_pos()
        if target_pos is not None:
            to_target = target_pos[:2] - pos[:2]
            target_angle = np.arctan2(to_target[1], to_target[0])
            if target_angle < 0:
                target_angle += 2 * np.pi
            for s in range(8):
                sector_angle = s * (2 * np.pi / 8)
                angle_diff = abs(target_angle - sector_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                # 高斯衰减：指向目标方向=1，偏离方向衰减
                self._cached_entropy_sectors[s] = np.exp(-2.0 * angle_diff)
            dist_to_target = np.linalg.norm(to_target)
            self._cached_dir_entropy = np.clip(1.0 - dist_to_target / 150.0, 0.0, 1.0)
        else:
            self._cached_entropy_sectors = np.ones(8) * 0.5
            self._cached_dir_entropy = 0.5

        obs[14:22] = self._cached_entropy_sectors * 2 - 1
        obs[22] = self._cached_dir_entropy * 2 - 1

        # 6. 定位不确定性(1): 阶段1简化为0
        obs[23] = 0.0

        return np.clip(obs, -1.0, 1.0)

    def _get_target_distance(self) -> float:
        """获取到当前目标的距离"""
        pos = np.array(self.uav_state['pos'])

        if self.current_task == self.TASK_TURBINE and len(self.turbines) > 0:
            return np.linalg.norm(pos - self.turbines[0].position)
        elif self.current_task == self.TASK_CABLE and len(self.cables) > 0:
            _, dist, _ = self.cables[0].get_nearest_point(pos)
            return dist
        elif self.current_task == self.TASK_RETURN:
            return np.linalg.norm(pos - self.base_position)
        else:
            return 0.0

    def _get_current_target_pos(self) -> np.ndarray:
        """获取当前目标位置"""
        if self.current_task == self.TASK_TURBINE and len(self.turbines) > 0:
            return self.turbines[0].get_hub_position()
        elif self.current_task == self.TASK_CABLE and len(self.cables) > 0:
            # 电缆中点
            pts = self.cable_points_list[0]
            return pts[len(pts) // 2]
        elif self.current_task == self.TASK_RETURN:
            return self.base_position
        else:
            return None

    def _get_info(self) -> dict:
        """构建info字典"""
        return {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "coverage": self.overall_coverage,
            "turbine_coverage": self.turbine_coverage,
            "cable_coverage": self.cable_coverage,
            "battery": self.battery.battery,
            "collision_count": self.collision_count,
            "current_task": self.current_task,
            "pos": self.uav_state['pos'],
        }


if __name__ == "__main__":
    print("=== InspectionEnv 自测 ===\n")

    env = InspectionEnv()

    # 1. reset测试
    obs, info = env.reset(seed=42)
    print(f"reset后obs形状: {obs.shape}")
    assert obs.shape == (24,), f"obs形状错误: {obs.shape}"
    print(f"reset后obs范围: [{obs.min():.3f}, {obs.max():.3f}]")
    assert obs.min() >= -1.0 and obs.max() <= 1.0, "obs超出[-1,1]"
    print(f"info: step={info['step']}, coverage={info['coverage']:.3f}, battery={info['battery']:.3f}")

    # 2. step测试
    action = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # 不修正，不切换任务
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"\nstep后obs形状: {obs2.shape}")
    print(f"reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
    print(f"info: step={info2['step']}, coverage={info2['coverage']:.3f}, "
          f"battery={info2['battery']:.3f}, collision={info2['collision_count']}")
    assert obs2.shape == (24,), "step后obs形状错误"

    # 3. 连续运行测试
    env.reset(seed=42)
    total_reward = 0.0
    total_coverage = 0.0
    for i in range(100):
        # 随机动作（残差=0，只走规则控制器）
        action = np.zeros(5, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_coverage = max(total_coverage, info['coverage'])
        if terminated or truncated:
            print(f"\nepisode在第{i+1}步结束: terminated={terminated}, truncated={truncated}")
            break
    print(f"\n100步测试: total_reward={total_reward:.2f}, max_coverage={total_coverage:.3f}")
    print(f"最终位置: {info['pos']}")
    print(f"最终电量: {info['battery']:.3f}")

    # 4. 观测空间兼容性测试
    env.reset(seed=100)
    for i in range(10):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs), f"第{i}步obs不在观测空间内"

    # 5. 覆盖完成测试（模拟快速覆盖）
    env.reset(seed=42)
    env.uav_state['pos'] = [50.0, 0.0, 35.0]  # 放在风机旁边
    for i in range(50):
        obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
        if terminated:
            print(f"\n覆盖完成! 步数={i+1}, coverage={info['coverage']:.3f}")
            break

    # 6. 低电量返航测试
    env.reset(seed=42)
    env.battery.battery = 0.1  # 强制低电量
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    print(f"\n低电量时任务: {info['current_task']} (应为3=返航)")
    assert info['current_task'] == 3, "低电量应强制返航"

    print("\n=== 所有测试通过 ===")
