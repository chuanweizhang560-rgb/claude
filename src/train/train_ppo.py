"""RLlib PPO训练脚本：单智能体残差RL

使用Ray RLlib的PPO算法训练InspectionEnv。
规则控制器提供引导速度，RL学习残差修正。
"""

import sys
import os

# 确保项目根目录在sys.path中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.envs.inspection_env import InspectionEnv
from src.configs.world_config import get_phase1_config


def env_creator(env_config):
    """RLlib环境创建函数"""
    # 使用world_config作为基础，env_config可覆盖部分参数
    config = get_phase1_config()
    if env_config:
        for k, v in env_config.items():
            config[k] = v
    return InspectionEnv(config=config)


def train(num_iterations: int = 200, checkpoint_freq: int = 20):
    """训练PPO

    Args:
        num_iterations: 训练迭代数（每次=train_batch_size步采样+1次更新）
        checkpoint_freq: 每N次迭代保存一次checkpoint
    """
    # 注册自定义环境
    register_env("inspection", env_creator)

    # 初始化Ray（限制资源避免OOM）
    ray.init(ignore_reinit_error=True, num_cpus=2, object_store_memory=500_000_000)

    # PPO配置（RLlib 2.55+ 新API）
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="inspection",
            env_config={},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length="auto",
        )
        .training(
            lr=3e-4,
            train_batch_size=2000,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            num_epochs=5,
            minibatch_size=64,
        )
        .resources(
            num_gpus=0,  # CPU训练即可，小模型不需要GPU
        )
        .debugging(
            log_level="WARN",
        )
    )

    # 构建算法
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        algo = config.build_algo()

    # 训练循环
    print(f"开始PPO训练: {num_iterations} iterations, "
          f"train_batch_size={config.train_batch_size}")
    print(f"检查点保存到: outputs/checkpoints/")
    print(f"TensorBoard日志到: outputs/tb_logs/")
    print("-" * 60)

    best_reward = -float("inf")

    for i in range(num_iterations):
        result = algo.train()

        # 提取关键指标（RLlib 2.55 新result结构）
        env_runners = result.get("env_runners", {})
        episode_reward_mean = env_runners.get("episode_reward_mean",
                            result.get("episode_reward_mean", 0.0))
        episode_len_mean = env_runners.get("episode_len_mean",
                          result.get("episode_len_mean", 0.0))

        # 获取loss（兼容不同result格式）
        learner_info = result.get("info", {}).get("learner", {})
        policy_info = learner_info.get("default_policy", learner_info.get("policy_0", {}))
        policy_loss = policy_info.get("policy_loss", policy_info.get("total_loss", 0.0))
        vf_loss = policy_info.get("vf_loss", 0.0)

        # 每10次迭代打印一次
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Iter {i+1:4d}/{num_iterations} | "
                  f"reward_mean={episode_reward_mean:8.2f} | "
                  f"ep_len={episode_len_mean:6.0f} | "
                  f"policy_loss={policy_loss:8.4f} | "
                  f"vf_loss={vf_loss:8.4f}")

        # 保存最优checkpoint
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_dir = algo.save(
                os.path.join(_project_root, "outputs", "checkpoints", "best")
            )
            if (i + 1) % 50 == 0:
                print(f"  → 新最优! 保存到 {checkpoint_dir}")

        # 定期保存checkpoint
        if (i + 1) % checkpoint_freq == 0:
            checkpoint_dir = algo.save(
                os.path.join(_project_root, "outputs", "checkpoints", f"iter_{i+1}")
            )
            print(f"  → 检查点已保存: {checkpoint_dir}")

    # 最终保存
    final_dir = algo.save(os.path.join(_project_root, "outputs", "checkpoints", "final"))
    print(f"\n训练完成! 最终checkpoint: {final_dir}")
    print(f"最优平均奖励: {best_reward:.2f}")

    algo.stop()
    ray.shutdown()

    return final_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO训练")
    parser.add_argument("--iters", type=int, default=200, help="训练迭代数")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="检查点保存频率")
    args = parser.parse_args()

    train(num_iterations=args.iters, checkpoint_freq=args.checkpoint_freq)
