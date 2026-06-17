"""世界配置：场景参数、任务定义、UAV初始状态"""

import numpy as np


def get_phase1_config() -> dict:
    """阶段1：单机基础课程（1风机 + 1电缆）"""

    return {
        # === 场景 ===
        "world_size": 300.0,           # 世界尺寸 m

        # === 风机 ===
        "turbines": [
            {
                "position": np.array([50.0, 0.0, 0.0]),
                "radius": 1.5,
                "height": 70.0,
            }
        ],

        # === 电缆 ===
        "cables": [
            {
                "start": np.array([0.0, 0.0, 30.0]),
                "end": np.array([100.0, 0.0, 30.0]),
                "sag": 4.0,
                "n_points": 50,        # 采样点数
            }
        ],

        # === 基站 ===
        "base_position": np.array([0.0, 0.0, 0.0]),
        "base_radius": 5.0,

        # === UAV ===
        "uav_start_pos": np.array([40.0, 10.0, 10.0]),  # 接近风机
        "uav_start_vel": np.array([0.0, 0.0, 0.0]),
        "uav_start_yaw": 0.0,
        "uav_start_battery": 1.0,
        "uav_role": "primary",         # 阶段1固定primary
        "uav_group_id": 0,

        # === 风场 ===
        "wind": {
            "steady_speed": 3.0,
            "steady_direction": 0.0,    # rad，沿x轴
            "turbulence_intensity": 0.3,
            "turbulence_freq": 1.0,
            "gust_enabled": False,
        },

        # === 地形 ===
        "terrain": {
            "size": 300.0,
            "max_height": 8.0,
            "resolution": 257,
            "seed": 42,
        },

        # === 动力学 ===
        "dynamics": {
            "kp": 0.6452,
            "kd": 0.01,
            "kp_z": 1.57,
            "kd_z": 0.01,
            "max_vel": 5.0,
            "max_vel_z": 3.0,
            "max_yaw_rate": 1.0,
            "dt": 0.05,
        },

        # === 碰撞 ===
        "collision": {
            "safe_margin": 1.5,
            "uav_safe_dist": 3.0,
            "ground_margin": 0.5,
            "clip_ratio": 0.3,
        },

        # === 覆盖 ===
        "coverage": {
            "coverage_dist_turbine": 12.0,
            "coverage_dist_cable": 5.0,
            "coverage_threshold": 0.85,
        },

        # === OctoMap ===
        "octomap": {
            "resolution": 0.5,
            "origin": np.array([-150.0, -150.0, 0.0]),
            "size": np.array([300, 300, 50]),
        },

        # === 电池 ===
        "battery": {
            "low_threshold": 0.15,
        },

        # === RL动作缩放 ===
        "action_scale": {
            "horizontal": 1.0,    # Δvx,Δvy 缩放：±1 → ±1 m/s
            "vertical": 0.5,     # Δvz 缩放：±1 → ±0.5 m/s
            "yaw_rate": 0.5,     # Δyaw_rate 缩放：±1 → ±0.5 rad/s
        },

        # === Episode ===
        "max_steps": 1000,           # 最大步数（1000*0.05=50s仿真时间）
        "randomize_start": True,     # 随机偏移起始位置
        "start_pos_range": 3.0,      # 起始位置随机范围 ±3m（减少初始距离差异）
        "randomize_wind": True,      # 随机风向偏移

        # === 奖励权重 ===
        "reward_weights": {
            "coverage": 10.0,
            "collision": 5.0,
            "power": 0.1,
            "battery_dead": 20.0,
            "coverage_bonus": 50.0,   # 全覆盖完成额外奖励
        },
    }


def get_phase2_config() -> dict:
    """阶段2：双机协同（2风机 + 1电缆）"""
    config = get_phase1_config()
    config["turbines"] = [
        {"position": np.array([50.0, 0.0, 0.0]), "radius": 1.5, "height": 70.0},
        {"position": np.array([150.0, 50.0, 0.0]), "radius": 1.5, "height": 70.0},
    ]
    config["uav_start_pos"] = np.array([5.0, 5.0, 5.0])
    return config


def get_phase3_config() -> dict:
    """阶段3：四机+约束（4风机 + 2电缆）"""
    config = get_phase1_config()
    config["turbines"] = [
        {"position": np.array([50.0, 0.0, 0.0]), "radius": 1.5, "height": 70.0},
        {"position": np.array([150.0, 50.0, 0.0]), "radius": 1.5, "height": 70.0},
        {"position": np.array([100.0, 100.0, 0.0]), "radius": 1.5, "height": 70.0},
        {"position": np.array([200.0, 80.0, 0.0]), "radius": 1.5, "height": 70.0},
    ]
    config["cables"] = [
        {"start": np.array([0.0, 0.0, 30.0]), "end": np.array([100.0, 0.0, 30.0]), "sag": 4.0, "n_points": 50},
        {"start": np.array([100.0, 50.0, 30.0]), "end": np.array([200.0, 80.0, 30.0]), "sag": 3.0, "n_points": 50},
    ]
    return config
