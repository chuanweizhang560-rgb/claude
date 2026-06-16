#!/usr/bin/env bash
set -euo pipefail

# ZCW项目PX4+Gazebo启动脚本
# 用法: bash scripts/start_px4_sim.sh [world_name]

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PROJECT_DIR}/assets/px4"
WORLD_NAME="${1:-}"

cd "${PX4_DIR}"

# 设置环境
source /opt/ros/humble/setup.bash 2>/dev/null || true

# 设置world
if [[ -n "${WORLD_NAME}" ]]; then
    export PX4_SITL_WORLD="${WORLD_NAME}"
fi

# 启动PX4 SITL + Gazebo Classic (headless)
echo "启动PX4 SITL + Gazebo Classic..."
echo "PX4目录: ${PX4_DIR}"
echo "World: ${WORLD_NAME:-default}"

HEADLESS=1 make px4_sitl gazebo-classic
