# 部署到训练机（UCloud node2，2026-08-05 实测）

## 1. uni_agent_ext 扩展包

```bash
# 本地打包上传
tar -czf /tmp/uni_agent_ext.tgz uni_agent_ext
# 训练机上解压到 /home/ubuntu/uni_agent_ext，并加 .pth 进 PYTHONPATH：
cd /home/ubuntu && tar -xzf /tmp/uni_agent_ext.tgz
echo "/home/ubuntu/uni_agent_ext" > \
  /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/uni_agent_ext.pth
```

> ⚠️ **.pth 内容必须是包的父目录**（`/home/ubuntu`），不是包目录本身
> （`/home/ubuntu/uni_agent_ext`）——否则 `import uni_agent_ext` 会去找
> `/home/ubuntu/uni_agent_ext/uni_agent_ext/__init__.py`，报 No module named
> （v0.6.0 实测踩坑）。

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
