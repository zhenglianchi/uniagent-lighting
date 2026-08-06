# 部署到训练机（UCloud node2，2026-08-05 实测）

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
for f in fix_multinode_hosts.sh nccl_multinode_test.py patch_verl_ipc_cpu.py \
         ray_import_test.py reward_smoke.py run_grpo_multinode_ucloud.sh \
         run_grpo_single_agentic_ucloud.sh run_grpo_single_lora_ucloud.sh \
         run_grpo_smoke_ucloud.sh; do
  rm -f "$f" && ln -s /home/ubuntu/uniagent-lighting/scripts/$f "$f"
done
```

更新流程：本地改代码 → commit + push（语义化版本）→ 服务器 `git -C /home/ubuntu/uniagent-lighting pull`。
`/home/ubuntu/swe-rl` 只放非仓库内容：`tencent_sandbox.env`（凭据，勿入库）、`data/`、`checkpoints/`、`logs/`。

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

## 6. mini-swe-agent tencent_e2b attach 补丁（训练机必做）

训练机安装的 `mini-swe-agent` 里的 `tencent_e2b` 环境类需两处改动：

1. `TencentE2BEnvironmentConfig` 增加 `attach_instance_id`；`_create()` 有 attach id 时
   直接 `Sandbox.connect`（跳过 StartSandboxInstance）——可整文件覆盖自
   `mini-swe-agent/src/minisweagent/environments/extra/tencent_e2b.py`
2. `cleanup()` 在 attach 模式下**只断开、不 kill/不停实例**（生命周期归 runner，
   reward 评估还要用）——否则 mini-extra 退出会停实例，reward 写 test_patch 报
   "The requested resource does not exist"（v0.13.1 实测踩坑）
