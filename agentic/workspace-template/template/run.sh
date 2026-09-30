#!/usr/bin/env bash
# run.sh — 启动 Pi 音频分析 workspace（issue #31，git-bash / POSIX）。
#
# 用法（在 workspace 目录）：
#   ./run.sh                       # 交互式 Pi
#   ./run.sh "分析 audio/inputs/fixture-a.wav"   # 带初始提示
#
# 作用同 run.ps1：设置 PI_AUDIO_BRIDGE_ROOT、SAM_AUDIO_ROOT，并以 `pi --approve`
# 启动（进程级 project trust；不写 ~/.pi/agent/trust.json、不改全局设置/认证）。
set -euo pipefail
cd "$(dirname "$0")"

export PI_AUDIO_BRIDGE_ROOT="$(pwd)/audio/inputs"

if [ -f local/config.json ]; then
    SAM_ROOT_CFG=$(python -c "import json;print(json.load(open('local/config.json',encoding='utf-8')).get('sam_root') or '')" 2>/dev/null || true)
    case "$SAM_ROOT_CFG" in
        ""|\<*) : ;;
        *) export SAM_AUDIO_ROOT="$SAM_ROOT_CFG" ;;
    esac
fi

# 模型由 .pi/settings.json 固定为 google/gemini-3.8-flash（不要用 --model 覆盖）。
exec pi --approve "$@"
