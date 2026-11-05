#!/usr/bin/env bash
# 机票退改签 Agent：安装依赖并启动。有 app.py 时启动 Web 工作台，否则进入命令行对话。
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "请先安装 uv：https://docs.astral.sh/uv/"; exit 1; }
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
uv sync
if [ -f app.py ]; then
  exec uv run python app.py "$@"
else
  exec uv run python -m air_agent "$@"
fi
