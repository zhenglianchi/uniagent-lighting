# 部署到训练机（UCloud node1，2026-08-08 实测；当前 117.50.189.37，1×4090 48G / 94G）

> 版本链（2026-08-04 起实测）：**torch 2.9.0+cu128 / vllm 0.11.1 / transformers 4.57.x /
> verl 0.9.0.dev（uni-agent 捆绑）/ ray 2.56.1 / TransferQueue 0.1.9**；
> vllm ≥0.11.1 为多机硬性要求。node2 不单独装环境，克隆 node1 镜像。

## 0. 仓库管理（2026-08-06 起，git 化）

服务器上的改造代码统一由 `uniagent-lighting` 仓库管理，不再手动传文件：

```bash
# node2 上（一次性）
cd /home/ubuntu
git clone https://github.com/zhenglianchi/uniagent-lighting.git

# uni_agent_ext -> 仓库软链（保留旧拷贝备份）
[ -L uni_agent_ext ] || mv uni_agent_ext uni_agent_ext.bak-20260806
ln -sfn /home/ubuntu/uniagent-lighting/uni_agent_ext uni_agent_ext

# swe-rl 运行目录（数据/凭据/checkpoint/日志）不动，脚本替换为仓库软链
cd /home/ubuntu/swe-rl
for f in run_grpo_smoke_ucloud.sh run_grpo_single_lora_ucloud.sh \
         run_grpo_single_agentic_ucloud.sh run_grpo_humanevalfix_ucloud.sh \
         run_grpo_multinode_ucloud.sh run_grpo_multinode_async_ucloud.sh \
         spec_train_run.sh kill_train.sh start_stats_watch.sh collect_grpo_stats.py \
         eval_humanevalfix.py convert_verl_lora_to_hf.py fix_multinode_hosts.sh \
         nccl_multinode_test.py patch_verl_ipc_cpu.py ray_import_test.py \
         reward_smoke.py tencent_stop_all_instances.py kill_eval.sh \
         eval_spec_final.sh run_eval_only.sh run_eval_final_spec.sh; do
  rm -f "$f" && ln -s /home/ubuntu/uniagent-lighting/scripts/$f "$f"
done
```

更新流程：本地改代码 → commit + push（语义化版本）→ 服务器 `git -C /home/ubuntu/uniagent-lighting pull`。
`/home/ubuntu/swe-rl` 只放非仓库内容：`tencent_sandbox.env`（凭据，勿入库）、`data/`、`checkpoints/`、`logs/`。

> 服务器侧 verl/uni-agent 补丁（部署时应用）：
> - py3.10 StrEnum 兼容：`scripts/fix_strenum_ucloud.sh`
> - 单卡 fsdp2 跳过冗余 state_dict 拷贝：内嵌于 setup 脚本（幂等）
> - IPC CPU 大权重：`scripts/patch_verl_ipc_cpu.py`（bucket 2048 + 发送前 CPU→CUDA）
> - `patches/verl_vllm_logprobs_spec_fix.patch`（EAGLE-3 下 logprobs 0 全丢，0→1）
> - `patches/verl_merged_lora_materialize_fix.patch`（merge=True 同步基座权重 bug，
>   backport verl#7014；服务器另可用 `scripts/patch_verl_merged_lora.py`）
> - `patches/verl_debug_metrics_logprobs_guard.patch`（batch 缺 rollout_log_probs 防崩）
> - `patches/gateway_hermes_parse_guard.patch`（解析容错，git apply + commit `5cc88ec`；
>   含 import 修复，覆盖 `patches/uni_agent_vllm0111_toolparsers.patch` 的 import 段）

> 历史方式（v0.18.0 前）：本地 tar 打包 uni_agent_ext → 服务器解压 → 写 `.pth` 进 PYTHONPATH。
> `.pth` 内容必须是包的父目录（`/home/ubuntu`），不是包目录本身——否则 `import uni_agent_ext`
> 会去找 `/home/ubuntu/uni_agent_ext/uni_agent_ext/__init__.py`，报 No module named（v0.6.0 实测踩坑）。

## 1. uni_agent_ext 扩展包

```bash
# 本地打包上传
tar -czf /tmp/uni_agent_ext.tgz uni_agent_ext
# 训练机上解压到 /home/ubuntu/uni_agent_ext，并加 .pth 进 PYTHONPATH：
cd /home/ubuntu && tar -xzf /tmp/uni_agent_ext.tgz
echo "/home/ubuntu/uni_agent_ext" > \
  /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/uni_agent_ext.pth
```

> ⚠️ git 化后不再需要本节（软链方式）；`.pth` 已在环境中，软链解析后 `import uni_agent_ext` 正常。

## 1b. 沙箱 SDK（runner 创建腾讯沙箱需要）

```bash
pip install "e2b-code-interpreter==2.9.0" tencentcloud-sdk-python-ags
# 注意：ags 模块版本路径随 SDK 版本变化（3.1.149 是 tencentcloud.ags.v20250920）
```

## 2. uni-agent Python 3.10 兼容补丁

uni-agent 一处误用 `typing.NotRequired`（3.11+ 才有）→ 3.10 下 import 失败。
打 `patches/uni_agent_py310_compat.patch`（或手动改
`uni_agent/gateway/adapters/types.py` 用 typing_extensions 回退）。

## 3. 数据与凭据

```bash
# agentic 训练数据（make_agentic_data.py 产物）
scp work/data/agentic_train.jsonl work/data/agentic_val.jsonl \
  ubuntu@<训练机IP>:/home/ubuntu/swe-rl/data/
# HumanEvalFix 数据（当前训练口径；train161 + val + smoke）
scp work/data/humanevalfix_train161.jsonl work/data/humanevalfix_val.jsonl \
  ubuntu@<训练机IP>:/home/ubuntu/swe-rl/data/
# 腾讯云沙箱凭据（chmod 600，勿入库）
scp work/tencent_sandbox.env ubuntu@<训练机IP>:/home/ubuntu/swe-rl/
```

## 4. 验证

```bash
python -c "import uni_agent_ext; \
from uni_agent_ext.agents.mini_swe_agent_runner import mini_swe_agent_runner; \
print(mini_swe_agent_runner.__name__)"
```

## 5. 运行 agentic 训练

`bash /home/ubuntu/swe-rl/run_grpo_single_agentic_ucloud.sh`（脚本见 scripts/）。
注意：腾讯沙箱需能访问训练机 Gateway（公网端口，见 docs/vllm_access.md）。

全样本口径（当前主力）：`bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_ucloud.sh`
（train161 / batch32 / mini16 / micro4 / 并发 64 / 5 epoch）；投机解码加
`bash /home/ubuntu/swe-rl/spec_train_run.sh`（`lora.merge=True` + EAGLE-3，独立
checkpoint/日志目录）。双机全异步：`run_grpo_multinode_async_ucloud.sh`（见 README）。

## 6. mini-swe-agent tencent_e2b attach 补丁（训练机必做）

训练机安装的 `mini-swe-agent` 里的 `tencent_e2b` 环境类需两处改动：

1. `TencentE2BEnvironmentConfig` 增加 `attach_instance_id`；`_create()` 有 attach id 时
   直接 `Sandbox.connect`（跳过 StartSandboxInstance）——可整文件覆盖自
   `mini-swe-agent/src/minisweagent/environments/extra/tencent_e2b.py`
2. `cleanup()` 在 attach 模式下**只断开、不 kill/不停实例**（生命周期归 runner，
   reward 评估还要用）——否则 mini-extra 退出会停实例，reward 写 test_patch 报
   "The requested resource does not exist"（v0.13.1 实测踩坑）

另外 pip 官方版 `minisweagent/run/benchmarks/swebench.py` 的镜像注入列表只有
`["docker", "swerex_modal"]`，**不含 `tencent_e2b`**，需覆盖为补丁版
（`patches/miniswe_swebench.py`），否则腾讯沙箱实例启动时 image 为空、无 `/testbed`
（v0.20.1 实测踩坑）。
