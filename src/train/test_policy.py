"""策略测试脚本：加载PPO checkpoint运行评估

运行1个episode，打印指标并生成轨迹图。
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ray.rllib.algorithms.ppo import PPO
from src.envs.inspection_env import InspectionEnv
from src.configs.world_config import get_phase1_config


def test_policy(checkpoint_path: str = None, num_episodes: int = 1, render: bool = True):
    """加载checkpoint运行评估

    Args:
        checkpoint_path: checkpoint目录路径，None则使用随机策略
        num_episodes: 评估episode数
        render: 是否生成轨迹图
    """
    # 创建环境
    config = get_phase1_config()
    env = InspectionEnv(config=config)

    # 加载策略
    if checkpoint_path:
        from ray import tune
        from ray.tune.registry import register_env

        def env_creator(env_config):
            cfg = get_phase1_config()
            return InspectionEnv(config=cfg)

        register_env("inspection", env_creator)

        import ray
        ray.init(ignore_reinit_error=True)

        algo = PPO.from_checkpoint(checkpoint_path)
        print(f"已加载checkpoint: {checkpoint_path}")
    else:
        algo = None
        print("使用随机策略")

    all_results = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0
        steps = 0
        done = False

        # 记录轨迹
        positions = [info['pos'].copy() if isinstance(info['pos'], list) else list(info['pos'])]
        coverages = [info['coverage']]
        batteries = [info['battery']]
        rewards = []

        while not done:
            if algo:
                # 使用训练好的策略
                action = algo.compute_single_action(obs)
            else:
                # 随机策略
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            positions.append(info['pos'].copy() if isinstance(info['pos'], list) else list(info['pos']))
            coverages.append(info['coverage'])
            batteries.append(info['battery'])
            rewards.append(reward)

        result = {
            "episode": ep,
            "total_reward": total_reward,
            "steps": steps,
            "final_coverage": info['coverage'],
            "turbine_coverage": info['turbine_coverage'],
            "cable_coverage": info['cable_coverage'],
            "final_battery": info['battery'],
            "collision_count": info['collision_count'],
            "terminated": terminated,
        }
        all_results.append(result)

        print(f"\nEpisode {ep+1}/{num_episodes}:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  步数: {steps}")
        print(f"  总覆盖: {info['coverage']:.3f}")
        print(f"  风机覆盖: {info['turbine_coverage']}")
        print(f"  电缆覆盖: {info['cable_coverage']}")
        print(f"  最终电量: {info['battery']:.3f}")
        print(f"  碰撞次数: {info['collision_count']}")
        print(f"  Termination: {'Coverage done' if terminated else 'Timeout/Battery dead'}")

        # 生成轨迹图
        if render:
            _plot_trajectory(positions, coverages, batteries, rewards, config, ep)

    if algo:
        algo.stop()
        import ray
        ray.shutdown()

    return all_results


def _plot_trajectory(positions, coverages, batteries, rewards, config, episode):
    """生成轨迹可视化图"""
    positions = np.array(positions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. XY轨迹 + 风机 + 电缆
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6, linewidth=0.5, label='UAV轨迹')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='起点')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=8, label='终点')

    # 风机
    for t_cfg in config.get("turbines", []):
        t_pos = t_cfg["position"]
        circle = Circle((t_pos[0], t_pos[1]), 1.5, color='orange', alpha=0.5)
        ax.add_patch(circle)
        ax.annotate('Turbine', (t_pos[0], t_pos[1]), ha='center', fontsize=8)

    # 电缆
    for c_cfg in config.get("cables", []):
        cable_x = [c_cfg["start"][0], c_cfg["end"][0]]
        cable_y = [c_cfg["start"][1], c_cfg["end"][1]]
        ax.plot(cable_x, cable_y, 'r--', alpha=0.5, label='Cable')

    # 基站
    base = config.get("base_position", [0, 0, 0])
    circle = Circle((base[0], base[1]), config.get("base_radius", 5), color='green', alpha=0.3)
    ax.add_patch(circle)
    ax.annotate('Base', (base[0], base[1]), ha='center', fontsize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 2. XZ轨迹（高度剖面）
    ax = axes[0, 1]
    steps = np.arange(len(positions))
    ax.plot(steps, positions[:, 2], 'b-', alpha=0.6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Altitude Z (m)')
    ax.set_title('Altitude Profile')
    ax.grid(True, alpha=0.3)

    # 3. 覆盖进度
    ax = axes[1, 0]
    ax.plot(np.arange(len(coverages)), coverages, 'g-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage Progress')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. 电量
    ax = axes[1, 1]
    ax.plot(np.arange(len(batteries)), batteries, 'r-', linewidth=1.5)
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='低电量阈值')
    ax.set_xlabel('Step')
    ax.set_ylabel('Battery')
    ax.set_title('Battery Level')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Episode {episode + 1} Evaluation', fontsize=14)
    plt.tight_layout()

    # 保存
    output_path = os.path.join(_project_root, "outputs", "figures", f"policy_eval_ep{episode+1}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  轨迹图保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="策略测试")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint路径")
    parser.add_argument("--episodes", type=int, default=1, help="评估episode数")
    parser.add_argument("--no-render", action="store_true", help="不生成图")
    args = parser.parse_args()

    test_policy(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
    )
