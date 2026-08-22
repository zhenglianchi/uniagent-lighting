# scripts 目录

按功能分子目录整理（2026-08-22，v0.56.0）：

| 子目录 | 用途 | 脚本 |
|---|---|---|
| `train/` | 训练启动与编排 | `run_grpo_*_ucloud.sh`、`run_dual_formal_chain.sh`、`spec_train_run.sh` |
| `eval/` | 评测与产物分析 | `eval_*.sh`、`eval_humanevalfix.py`、`convert_verl_lora_to_hf.py`、`collect_grpo_stats.py`、`analyze_trajectory.py`、`reward_smoke.py` |
| `data/` | 数据集构建 | `make_*_data.py` |
| `sampling/` | 采样流水线 | `start_sampling.sh`、`trajectory_uploader.py` |
| `platform/` | 平台化本地 agent / 工具转发 | `platform_local_agent.py`、`platform_local_claude.py`、`sandbox_mcp_server.py` |
| `sandbox/` | 腾讯云沙箱接入 | `tencent_*.py`、`run_tencent_sandbox_demo.py`、`run_tencent_swebench_single.sh` |
| `ops/` | 集群 / 部署 / 运维 | `bootstrap_ray_env.sh`、`install_ucloud_from_scratch.sh`、`fix_*.sh`、`patch_verl_*.py`、`nccl_multinode_test.py`、`ssh_ucloud.py`、`proxy.sh`、`cc_connect.sh`、`upgrade_vllm_0111.sh` |
| `archive/` | 过时 / 一次性脚本归档 | 见 `archive/README.md` |

## 服务器部署约定

训练机（node1 / node2）仍按**扁平目录**部署：把运行所需脚本拷贝到
`/home/ubuntu/swe-rl/`（脚本之间按扁平相对路径互调），参考
`docs/deployment.md` §2.1。仓库内子目录路径用于本地阅读与文档引用。
