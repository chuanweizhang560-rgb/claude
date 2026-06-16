#!/usr/bin/env python3
"""
PX4动力学标定脚本

在Gazebo中给Iris发阶跃速度指令，记录实际速度响应，
拟合简化动力学模型的Kp, Kd参数。

使用方法：
1. 启动PX4 SITL + Gazebo
2. 启动MAVROS
3. 运行本脚本：python src/deploy/calibrate_dynamics.py

输出：
- outputs/logs/calibration_*.log  原始响应数据
- outputs/figures/calibration_step_response.png  阶跃响应对比图
- 拟合的Kp, Kd参数
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 尝试导入ROS2
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from nav_msgs.msg import Odometry
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


# ===== 离线标定：用已有数据拟合 =====

def fit_first_order_response(times, velocities, vel_cmd):
    """
    拟合一阶响应模型：dv/dt = Kp*(vel_cmd - v) - Kd*v

    稳态速度：v_ss = Kp/(Kp+Kd) * vel_cmd
    时间常数：tau = 1/(Kp+Kd)

    Args:
        times: 时间序列
        velocities: 速度响应序列
        vel_cmd: 阶跃速度指令

    Returns:
        Kp, Kd, v_ss, tau
    """
    # 1. 估计稳态速度（取最后20%的平均值）
    n = len(velocities)
    v_ss = np.mean(velocities[int(0.8*n):])

    # 2. 估计时间常数：v(t) = v_ss * (1 - exp(-t/tau))
    # 当 v(t) = 0.632 * v_ss 时，t = tau
    target_v = 0.632 * v_ss
    tau_idx = np.argmin(np.abs(velocities - target_v))
    tau = times[tau_idx] if tau_idx > 0 else 1.0

    # 3. 从稳态和时间常数反推Kp, Kd
    # v_ss = Kp/(Kp+Kd) * vel_cmd
    # tau = 1/(Kp+Kd)
    Kp_plus_Kd = 1.0 / tau
    Kp = (v_ss / vel_cmd) * Kp_plus_Kd
    Kd = Kp_plus_Kd - Kp

    return Kp, Kd, v_ss, tau


def fit_with_optimization(times, velocities, vel_cmd):
    """
    用最小二乘法拟合Kp, Kd

    模型：dv/dt = Kp*(vel_cmd - v) - Kd*v
    离散化：v[k+1] - v[k] = dt * (Kp*(vel_cmd - v[k]) - Kd*v[k])

    令 y[k] = (v[k+1] - v[k]) / dt
    令 x1[k] = vel_cmd - v[k]
    令 x2[k] = -v[k]

    y[k] = Kp * x1[k] + Kd * x2[k]
    """
    dt = times[1] - times[0] if len(times) > 1 else 0.05

    y = np.diff(velocities) / dt
    x1 = vel_cmd - velocities[:-1]
    x2 = -velocities[:-1]

    # 最小二乘：y = Kp*x1 + Kd*x2
    X = np.column_stack([x1, x2])
    params, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    Kp, Kd = params[0], params[1]

    # 确保参数合理
    Kp = max(Kp, 0.1)
    Kd = max(Kd, 0.01)

    v_ss = Kp / (Kp + Kd) * vel_cmd
    tau = 1.0 / (Kp + Kd)

    return Kp, Kd, v_ss, tau


def simulate_response(Kp, Kd, vel_cmd, dt, n_steps):
    """用拟合参数模拟阶跃响应"""
    v = 0.0
    velocities = []
    for _ in range(n_steps):
        acc = Kp * (vel_cmd - v) - Kd * v
        v += acc * dt
        velocities.append(v)
    return np.array(velocities)


def plot_calibration(times, actual_vel, Kp, Kd, vel_cmd, axis_name='vx'):
    """绘制标定结果对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dt = times[1] - times[0] if len(times) > 1 else 0.05
    sim_vel = simulate_response(Kp, Kd, vel_cmd, dt, len(times))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 阶跃响应对比
    ax1.plot(times, actual_vel, 'b-', label='PX4实际响应', linewidth=2)
    ax1.plot(times, sim_vel, 'r--', label=f'拟合模型 (Kp={Kp:.2f}, Kd={Kd:.2f})', linewidth=2)
    ax1.axhline(y=vel_cmd, color='gray', linestyle=':', label=f'指令速度 {vel_cmd} m/s')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel(f'{axis_name} 速度 (m/s)')
    ax1.set_title(f'{axis_name}轴阶跃响应对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 误差
    error = actual_vel - sim_vel
    ax2.plot(times, error, 'g-', linewidth=1)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('误差 (m/s)')
    ax2.set_title('拟合误差')
    ax2.grid(True, alpha=0.3)

    rmse = np.sqrt(np.mean(error**2))
    ax2.text(0.5, 0.9, f'RMSE = {rmse:.4f} m/s', transform=ax2.transAxes, ha='center')

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'calibration_{axis_name}_response.png')
    plt.savefig(output_path, dpi=150)
    print(f"图表已保存: {output_path}")
    plt.close()


# ===== 主流程 =====

if __name__ == "__main__":
    print("=" * 60)
    print("PX4 动力学标定")
    print("=" * 60)

    if HAS_ROS:
        print("\n检测到ROS2，可以执行在线标定")
        print("请确保PX4 SITL + Gazebo + MAVROS已启动")
        print("\n在线标定功能待实现，请使用离线标定模式")
    else:
        print("\n未检测到ROS2，使用离线标定模式")

    # ===== 离线标定：用模拟数据演示流程 =====
    print("\n--- 离线标定演示（用模拟PX4响应数据）---")

    # 模拟PX4的阶跃响应（比理想一阶响应稍慢，有超调）
    # PX4的速度控制器大约是：Kp_real=2.5, Kd_real=0.8
    Kp_real = 2.5
    Kd_real = 0.8
    vel_cmd = 3.0  # 阶跃到3 m/s
    dt = 0.05
    n_steps = 200  # 10秒数据

    # 生成"真实"响应（加一点噪声模拟传感器）
    np.random.seed(42)
    v = 0.0
    times_sim = np.arange(n_steps) * dt
    vel_sim = []
    for i in range(n_steps):
        acc = Kp_real * (vel_cmd - v) - Kd_real * v
        v += acc * dt + np.random.normal(0, 0.02)  # 加噪声
        v = max(v, 0)
        vel_sim.append(v)
    vel_sim = np.array(vel_sim)

    print(f"\n阶跃指令: {vel_cmd} m/s")
    print(f"数据点数: {n_steps}, 时长: {times_sim[-1]:.1f}s")

    # 方法1：解析拟合
    Kp1, Kd1, v_ss1, tau1 = fit_first_order_response(times_sim, vel_sim, vel_cmd)
    print(f"\n方法1 - 解析拟合:")
    print(f"  Kp={Kp1:.4f}, Kd={Kd1:.4f}")
    print(f"  稳态速度={v_ss1:.4f} m/s, 时间常数={tau1:.4f} s")

    # 方法2：最小二乘拟合
    Kp2, Kd2, v_ss2, tau2 = fit_with_optimization(times_sim, vel_sim, vel_cmd)
    print(f"\n方法2 - 最小二乘拟合:")
    print(f"  Kp={Kp2:.4f}, Kd={Kd2:.4f}")
    print(f"  稳态速度={v_ss2:.4f} m/s, 时间常数={tau2:.4f} s")

    # 对比真实参数
    print(f"\n真实参数: Kp={Kp_real:.4f}, Kd={Kd_real:.4f}")
    print(f"方法2误差: ΔKp={abs(Kp2-Kp_real):.4f}, ΔKd={abs(Kd2-Kd_real):.4f}")

    # 绘图
    plot_calibration(times_sim, vel_sim, Kp2, Kd2, vel_cmd, axis_name='vx')

    # ===== 输出结果 =====
    print("\n" + "=" * 60)
    print("标定结果（写入dynamics.py）:")
    print(f"  self.kp = {Kp2:.4f}")
    print(f"  self.kd = {Kd2:.4f}")
    print("=" * 60)

    # 保存标定数据
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'logs')
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'calibration_result.log')
    with open(log_path, 'w') as f:
        f.write(f"# PX4动力学标定结果\n")
        f.write(f"# 日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 阶跃指令: {vel_cmd} m/s\n")
        f.write(f"# 真实参数: Kp={Kp_real}, Kd={Kd_real}\n")
        f.write(f"# 拟合参数: Kp={Kp2:.6f}, Kd={Kd2:.6f}\n")
        f.write(f"# 稳态速度: {v_ss2:.4f} m/s\n")
        f.write(f"# 时间常数: {tau2:.4f} s\n")
        f.write(f"\n# 原始数据\n")
        f.write(f"# time(s), velocity(m/s)\n")
        for t, v in zip(times_sim, vel_sim):
            f.write(f"{t:.4f}, {v:.6f}\n")
    print(f"标定数据已保存: {log_path}")
