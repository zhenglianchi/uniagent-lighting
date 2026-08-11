# Roadmap（2026-08-06 定稿）

> 服务器已关机、node2 镜像已保存；所有代码/文档由本仓维护，恢复后
> `git -C /home/ubuntu/uniagent-lighting pull` 即可续跑。

## 1. 换数据集：HumanEvalFix（下一步，本地可做，agent 不改）

背景：Qwen3-8B 在 SWE-bench 长程任务上行为退化已实证（60 轮不修改代码、循环执行命令；
simple-bench 极简实验已回滚）。**用户拍板：agent 不改（保持 mini-swe-agent harness），
通过降低任务难度换取 8B 可出结果。**

- 数据集：`bigcode/humanevalpack` Python 修复子集——单函数 buggy 代码 + 单元测试，
  8B 级模型有公开 pass@1 基准（Granite 8B ≈ 25~48%），60 轮内可出结果
- ✅ 数据构造 `scripts/make_humanevalfix_data.py`（2026-08-06，原 SWE-bench
  `make_agentic_data.py` 保留不动）：humanevalpack python 子集 → solution.py +
  test_solution.py（check(candidate) → pytest 单测 `test_all`，`from solution import *`
  兼容同文件辅助函数）+ 本地 verify（buggy rc=1 / canonical rc=0，死循环超时自动跳过）
- ✅ 冒烟数据 `work/data/humanevalfix_train.jsonl`（3 条）+ `humanevalfix_val.jsonl`（2 条）入库
- ✅ runner 支持 `humaneval_fix` 任务类型（swe_bench 原路径不变）：沙箱 /testbed git
  仓库 + solution.py 注入（`git add -A`）+ mini-swe-agent API 直连 + reward 阶段写隐藏
  测试（无测试泄露）
- ✅ 训练脚本 `scripts/run_grpo_humanevalfix_ucloud.sh`（数据/实验名/checkpoint 目录区分）
- ⏳ 上机验证：node2 恢复镜像后 git pull → 数据拷到 `/home/ubuntu/swe-rl/data/` →
  跑 run_grpo_humanevalfix_ucloud.sh；验收 3~5 条样本至少 1 条修出通过补丁、reward 出现
  组内差异（advantage ≠ 0）→ 支撑完整训练链路，作为校招亮点（agentic 修复 + 沙箱系统）

## 2. 双机全异步 GRPO（2026-08-11 用户定稿：双机网络就绪后第一优先；PD 分离已放弃）

- 目标：双机（node1+node2）开启 verl v1 全异步——Trainer 与 rollout 重叠，摊平单机
  step 内 gen（53%）与 update（33%）的串行瓶颈（理论上限 ~1.8x）
- 脚本已备：`scripts/run_grpo_multinode_async_ucloud.sh`（v0.35.0）——默认
  `trainer.v1.trainer_mode=colocate_async`（rollout+trainer 同机重叠）；`separate_async`
  （训练机独立，需非 naive checkpoint engine 权重同步）实验性后测；TQ 已全程承载
  verl v1 数据流（SimpleStorage 默认）
- **Mooncake 不单跑**：无 RDMA 普通网卡 + 轨迹小数据量收益有限，不做双机 Mooncake
  对照实验
- 前提：两台同 VPC/子网（当前 node1/node2 不同 VPC 未通；node2 镜像已保存）
- 验收：step 墙钟 / 生成吞吐 vs 单机基线（投机 run 45.3min/step 为对照口径）

## 3. 投机解码（Speculative Decoding）

- 小 draft 模型（如 Qwen2.5-Coder-0.5B/1.5B）为 7B/8B 目标模型投机，接受率高时生成提速
  （vLLM 官方实测 1.5-3x）
- 落地点（待验证）：vllm `--speculative-config`（draft_model + num_speculative_tokens）
- 关键风险：RL 训练必须拿到目标分布精确 logprobs，vLLM 不保证 spec decode 下 logprob
  稳定性 → 需在 verl rollout 链路实测（distilled 类 draft 用 rejection sampling 保分布）
- 注意：draft 模型多占显存，适合双卡/多卡阶段；batch 大、KV 复用低时收益下降需实测阈值

## 4. 服务器恢复 checklist

1. 用户恢复 node2 镜像（新实例 ≥64GB 内存）→ 更新本地 `work/ucloud.env`（新公网/内网 IP）
2. node2：`git -C /home/ubuntu/uniagent-lighting pull`（本仓代码/文档/脚本）
3. 环境验证（import / 模型存在）→ 按 §1 构造 HumanEvalFix 数据 → 跑单机 agentic GRPO 冒烟
4. 双机网络就绪后：重做 hosts/SSH/Ray（`scripts/fix_multinode_hosts.sh` +
   `setup_ssh_trust.py` + `ray_cluster_setup.sh`）→ 跑双机全异步 GRPO
