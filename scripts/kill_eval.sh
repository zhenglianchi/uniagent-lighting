#!/usr/bin/env bash
# 清理残留的评测/服务进程（释放 GPU 显存）
set -u
pkill -9 -f "eval_humanevalfix" 2>/dev/null
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
sleep 3
echo "GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "PROCS=$(ps aux | grep -cE '[v]llm|[e]val_humanevalfix')"
