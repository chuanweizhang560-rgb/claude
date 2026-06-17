# 阶段1 Gymnasium环境封装 + PPO训练日志

**日期**: 2026-06-17
**状态**: ✅ 阶段1.1完成（Env封装+训练流程验证）

## 目标
将9个环境模块+4个规则控制器封装为标准Gymnasium Env，用RLlib PPO训练单智能体验证残差RL架构

## 实现内容

### 1.1 世界配置文件
- `src/configs/world_config.py`: 定义3个课程阶段配置
- 阶段1: 1风机(50,0,0) + 1电缆(0→100,y=0,z=30) + 基站(0,0,0)
- UAV起始: (40,10,10) 接近风机（加速训练初期获得覆盖奖励）
- max_steps=1000 (50s仿真时间)
- 奖励权重: coverage=10, collision=5, power=0.1, battery_dead=20, coverage_bonus=50

### 1.2 Gymnasium Env封装
- `src/envs/inspection_env.py`: InspectionEnv(gymnasium.Env)
- **观测空间24维**: pos_xyz(3)+vel_xyz(3)+battery(1)+yaw(1)+task_type(1)+target_dist(1)+coverage(1)+mask(1)+role(2)+8扇区方向编码(8)+目标方向熵(1)+定位不确定性(1)
- **动作空间5维**: [Δvx, Δvy, Δvz, Δyaw_rate, task_select] ∈ [-1,1]
- **残差RL核心**: `vel_cmd = rule_vel + delta_action`
- **规则控制器切换**: task_select → TurbineOrbit/CableFollow/ExploreGuide/ReturnHome
- **低电量强制返航**: battery<15%时覆盖task_select强制切ReturnHome

### 1.3 RLlib PPO训练
- `src/train/train_ppo.py`: 注册自定义Env + PPO训练循环
- 配置: lr=3e-4, train_batch=2000, num_epochs=5, minibatch=64, gamma=0.99
- 旧API stack (enable_rl_module_and_learner=False)
- 保存checkpoint到outputs/checkpoints/

### 1.4 策略评估脚本
- `src/train/test_policy.py`: 加载checkpoint/随机策略运行评估
- 生成4面板可视化: XY轨迹+高度+覆盖进度+电量

## 踩坑记录

### 问题1: RLlib 2.55 API变更
- **现象**: `rollouts()` deprecated, `sgd_minibatch_size`参数名变, result结构变
- **修复**: `rollouts()` → `env_runners()`, `num_sgd_iter` → `num_epochs`, 禁用新API stack, result中episode_reward_mean在`result['env_runners']`子字典

### 问题2: OctoMap扇区熵计算极慢
- **现象**: 300×300×50网格的get_entropy_sector每步耗时>1s
- **修复**: 阶段1跳过OctoMap更新，8扇区熵用目标方向高斯编码替代（指向目标=1，偏离衰减）
- **原理**: 阶段1是覆盖任务，不需要地图熵驱动探索

### 问题3: TurbineOrbitController环绕阶段UAV飞离风机
- **现象**: UAV距风机12.8m(在approach_dist=15m内)进入环绕阶段，但orbit_angle=0时切线方向让UAV飞离
- **修复**: 环绕阶段增加距离检查：距轨道点>5m时先飞向轨道点，<5m才用切线速度

### 问题4: 训练采样超时
- **现象**: "No samples returned from remote workers"
- **根因**: OctoMap熵计算使env step从0.6ms暴涨到>1s
- **修复**: 禁用OctoMap更新后每步0.6ms，采样正常

### 问题5: matplotlib中文缺字体
- **现象**: 图表中文字符显示为方框
- **修复**: 改用英文标注

## 训练结果

| Iter | reward_mean | ep_len |
|------|------------|--------|
| 1    | -452.6     | 884    |
| 10   | -387.4     | 883    |
| 20   | -362.0     | 883    |
| 30   | -362.0     | 883    |

- 30 iter (60K env steps)后reward从-452提升到-362
- 收敛趋势存在但缓慢，主要靠规则控制器引导
- reward全负数：功耗惩罚占主导，覆盖奖励不够大
- 需要更多迭代(200+)和reward shaping调整

## 文件清单
- `src/configs/__init__.py` + `world_config.py`
- `src/envs/inspection_env.py`
- `src/train/__init__.py` + `train_ppo.py` + `test_policy.py`
- `outputs/checkpoints/iter_10|20|30|best|final/`
- `outputs/figures/policy_eval_ep1.png`

## 待改进
- 增大覆盖奖励权重(10→50)或添加shape reward(接近目标距离递减奖励)
- 增加训练迭代到200+，观察收敛趋势
- 电量模型校准(悬停续航偏大)
- OctoMap训练优化：用向量化计算替代逐体素遍历
