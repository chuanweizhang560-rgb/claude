# 阶段0.2 规则控制器开发日志

**日期**: 2026-06-16
**状态**: ✅ 完成

## 目标
实现4个规则控制器，为残差RL提供引导速度

## 设计理念
- 规则控制器输出4维引导速度: [vx, vy, vz, yaw_rate]
- RL只学习残差修正: [Δvx, Δvy, Δvz, Δyaw_rate, task_select]
- 主/副角色: 同组内primary和assistant执行不同任务策略

## 控制器实现

### 1. TurbineOrbitController (风机巡检)
- **Primary**: 螺旋上升轨道, r=10m, 从底部螺旋到顶部
- **Assistant**: 悬停在hub高度, r=5m, 观测叶片
- 参数: orbit_radius=10, orbit_speed=0.5, orbit_height_min=2, orbit_height_max=80
- 关键方法: Frenet坐标系计算切线方向

### 2. CableFollowController (电缆巡检)
- **Primary**: 沿电缆Frenet坐标系前进, 保持正下方5m
- **Assistant**: 横移+5m lateral offset, 从侧面观测
- 参数: follow_speed=1.5, follow_height_offset=-5.0, lateral_offset=5.0
- 关键方法: 弧长参数化, Frenet切线/法线/副法线

### 3. ExploreGuideController (探索引导)
- 选择8扇区中熵值最高的方向前进
- 低于min_entropy_threshold=0.3时悬停
- 低空先爬升到explore_height=25m
- 扇区角度: 扇区0=0°(东), 1=45°, ..., 7=315°

### 4. ReturnHomeController (返航)
- 远处全速返航(return_speed=4), 近处减速(slow_radius=10)
- 到达基站悬停(hover_height=5)
- yaw指向基站方向

## 基类设计 (base_controller.py)
```python
class RuleController:
    def _vec2d(pos, target) -> 方向向量
    def _normalize_angle(angle) -> [-π, π]
    def _yaw_to_target(pos, target) -> 目标yaw
    def _yaw_rate_to_target(current_yaw, target_yaw, dt) -> yaw角速率
```

## 测试
每个控制器有`__main__`测试块:
- TurbineOrbit: 测试primary螺旋/assistant悬停
- CableFollow: 测试primary跟随/assistant横移
- ExploreGuide: 测试各方向/低空爬升/低熵悬停
- ReturnHome: 测试远处/近处/到达/高度调整

所有测试包含assert断言，运行通过。

## 接口约定
- 输入: state dict {pos, vel, yaw, targets, base_pos, entropy_map, ...}
- 输出: np.array([vx, vy, vz, yaw_rate])
- 不同控制器使用state中不同的字段
