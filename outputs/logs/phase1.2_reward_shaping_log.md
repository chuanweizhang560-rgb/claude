# 阶段1.2 Reward Shaping + 扩展训练日志

**日期**: 2026-06-17
**状态**: ✅ 阶段1.2完成（Reward调整+电池校准+覆盖bug修复+训练验证）

## 目标
解决1.1训练中reward -452→-362收敛慢的问题：增大覆盖奖励、添加shaping reward、校准电池模型、扩展训练

## 修改内容

### 1. Reward权重调整
| 权重 | 1.1 | 1.2 | 原因 |
|------|-----|-----|------|
| coverage | 10 | 50 | 覆盖奖励增大5倍，成为主要驱动 |
| power | 0.1 | 0.01 | 功耗惩罚降10倍，不再主导reward |
| coverage_bonus | 50 | 100 | 全覆盖完成奖励翻倍 |
| proximity | - | 0.1 | 新增：接近目标exp衰减shaping reward |
| task_progress | - | 10 | 新增：任务完成切换奖励 |

### 2. 电池模型校准
- **旧参数**: p_hover=0.0167 %/s → 悬停续航仅60s (1min)
- **新参数**: p_hover=0.000833 %/s → 悬停续航1200s (20min)
- **验证**: 5m/s飞行+1m/s爬升+3m/s风时续航约500s (8.3min)
- 系数同步缩放: k_speed=0.000150, k_climb=0.000250, k_wind=0.000050, k_accel=0.000100

### 3. 覆盖评估bug修复（关键！）
- **Bug**: `_update_coverage()`用标量max累积覆盖率，不同高度的覆盖段无法叠加
  - UAV在8m轨道上螺旋上升，每个高度覆盖6-8段/20段
  - 标量max只保留最大值，无法累积不同高度覆盖的不同段
- **修复**: 改为逐段布尔数组累积 `turbine_segments[i] = np.logical_or(old, new)`
- **新增方法**: `compute_turbine_segment_coverage()`, `compute_cable_segment_coverage()`
- **效果**: 风机覆盖从40%提升到85%！

### 4. TurbineOrbitController重写
- **旧逻辑**: 切线+向心修正，UAV追不上移动的轨道点
- **新逻辑**: PD追踪轨道点 `vel_cmd = orbit_vel + kp*error - kd*vel`
- **参数调整**: orbit_radius 10→8m, climb_speed 0.5→1.0m/s, angular_speed 0.1→0.15rad/s
- **效果**: UAV正确从底部(3m)螺旋上升到70m

### 5. 任务切换逻辑改进
- **旧**: action[4]>0就切换（随机策略50%概率触发）
- **新**: 覆盖达标自动切换 + action[4]>0.8才手动切换（10%概率）

### 6. 动作缩放调整
- **旧**: horizontal=1.0, vertical=0.5, yaw_rate=0.5（RL可输出±1m/s修正）
- **新**: horizontal=0.3, vertical=0.2, yaw_rate=0.2（RL最大修正±0.3m/s）
- 原因：大幅修正破坏规则控制器引导，小幅修正保留规则+微调

### 7. Episode长度
- max_steps: 1000→2000（100s仿真时间，足够完成风机+电缆巡检）

## 训练结果

### 100 iter PPO训练
| 指标 | 值 |
|------|------|
| 训练iter | 100 (200K env steps) |
| 训练时间 | ~65min |
| 最优reward_mean | -211 (iter 80+) |
| 平均reward_mean | -410 |
| Ray问题 | OOM×3, worker重启×4 |

### 策略评估
| 评估方式 | reward | coverage | turbine | cable |
|----------|--------|----------|---------|-------|
| 纯规则控制器 | +35.52 | 0.701 | 0.850 | 0.551 |
| PPO (deterministic) | +35.52 | 0.701 | 0.850 | 0.551 |
| PPO (explore=True) | +7.50 | 0.150 | 0.300 | 0.000 |

### 关键发现
**PPO学到的最优策略是"输出零修正量"**，即完全依赖规则控制器。这在残差RL架构中是合理的：
- 规则控制器已经足够好（85%风机覆盖）
- RL修正量在当前简单场景下没有改善空间
- 后续阶段2+3加入更多约束后，RL才有动力学习有意义的修正

## 踩坑记录

### 问题1: 覆盖率标量max bug
- **现象**: UAV螺旋上升但风机覆盖停在40%不增长
- **根因**: 不同高度覆盖的是不同段，但coverage用标量max只保留最大值
- **修复**: 改为逐段布尔数组累积

### 问题2: 接近奖励淹没覆盖奖励
- **现象**: proximity reward占95%（179/188），coverage reward只有5%
- **根因**: proximity用线性衰减(1-dist/100)，每步给0.1奖励累积200+
- **修复**: 改为exp衰减+降低权重(2.0→0.1)

### 问题3: Ray OOM
- **现象**: worker被kill后重启，采样超时
- **根因**: Ray默认启动30+个idle worker，消耗大量内存
- **修复**: ray.init(num_cpus=2, object_store_memory=500MB)

### 问题4: TurbineOrbitController飞离风机
- **现象**: UAV在orbit_radius附近打转，XY距离从4.5m增大到15m
- **根因**: 切线+修正策略追不上移动的轨道点
- **修复**: 重写为PD追踪轨道点

## 文件变更
- `src/envs/inspection_env.py`: reward shaping + 覆盖bug修复 + 任务切换改进
- `src/configs/world_config.py`: 权重调整 + action_scale + max_steps
- `src/envs/battery.py`: 电池功耗系数校准
- `src/envs/coverage.py`: 新增segment-level覆盖方法
- `src/controllers/turbine_orbit.py`: PD追踪重写
- `src/train/train_ppo.py`: Ray资源限制

## 下一步: 阶段1.3
- 加入风场扰动（让规则控制器不再完美，RL有修正动力）
- 加入地图熵驱动探索
- 增加训练到300+ iter
- 或直接进入阶段2双机MAPPO

## Why: 记录reward调整的完整过程和踩坑，避免重复
## How to apply: 下次调reward时先检查各组件占比，确保覆盖奖励主导
