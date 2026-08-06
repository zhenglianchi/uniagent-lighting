# Roadmap（2026-08-06 定稿）

> 服务器已关机、node2 镜像已保存；所有代码/文档由本仓维护，恢复后
> `git -C /home/ubuntu/uniagent-lighting pull` 即可续跑。

## 1. 换数据集：HumanEvalFix（下一步，本地可做，agent 不改）

背景：Qwen3-8B 在 SWE-bench 长程任务上行为退化已实证（60 轮不修改代码、循环执行命令；
simple-bench 极简实验已回滚）。**用户拍板：agent 不改（保持 mini-swe-agent harness），
通过降低任务难度换取 8B 可出结果。** 黑盒 agent（Claude Code 类）调研结论 = 可做但
决定不改（详见思路 1.10：Anthropic 官方 unsupported + ToS 灰色 + 格式桥工作量数天）。

- 数据集：`bigcode/humanevalpack` Python 修复子集——单函数 buggy 代码 + 单元测试，
  8B 级模型有公开 pass@1 基准（Granite 8B ≈ 25~48%），60 轮内可出结果
- 构造步骤（本地可做，无需 GPU）：
  1. hf-mirror 拉 `bigcode/humanevalpack`（`HF_HUB_DISABLE_XET=1`），过滤 Python 子集
  2. 每样本生成 `solution.py`（buggy 代码）+ `test_solution.py`（隐藏测试）+
     FAIL_TO_PASS / PASS_TO_PASS 清单
  3. 转 verl agentic 数据：`raw_prompt` + `tools_kwargs.task`（文件注入：预写
     solution.py + test_solution.py 到沙箱工作目录）+ `reward_model.ground_truth` +
     `ability`（沿用 `scripts/make_agentic_data.py` 的 schema）
  4. 冒烟 2 条 → 3~5 条验证 8B 通过率（预期 ≥50%）
- runner 改动：`uni_agent_ext/agents/mini_swe_agent_runner.py` 恢复/新增"任务文件注入"
  路径（建沙箱后先写 solution.py / test_solution.py 再启动 agent）；reward 沿用已通的
  pytest FAIL_TO_PASS 真实打分（无测试泄露：test_solution.py 只在 reward 阶段注入）
- 验收：3~5 条样本至少 1 条修出通过补丁 → 跑一轮 GRPO 观察 reward 出现组内差异
  （advantage ≠ 0）→ 支撑完整训练链路，作为校招亮点（agentic 修复 + 沙箱系统）

## 2. 双机 TQ + Mooncake（双机网络就绪后第一优先）

- 目标：双机（node1+node2）开启 verl TransferQueue 跨节点轨迹缓冲 + Mooncake
  （KV cache 传输 = MooncakeConnector；P2P 权重分发 = mooncake p2p store /
  checkpoint-engine），摊平单机 rollout 与权重同步瓶颈
- 前提：两台同 VPC/子网（当前 node1/node2 不同 VPC 未通；node2 镜像已保存）
- 落地点（待上机验证）：TQ 后端（SimpleStorage 双机 or Redis）→ 双机 GRPO 跑通 →
  vllm PD 分离场景 KV 走 MooncakeConnector；权重分发走 mooncake P2P
- 验收：跨节点生成吞吐 / KV 传输延迟 / 每步权重同步耗时 vs 单机基线；
  普通款无 RDMA 时收益待实测

## 3. 投机解码（Speculative Decoding）

- 小 draft 模型（如 Qwen2.5-Coder-0.5B/1.5B）为 7B/8B 目标模型投机，接受率高时生成提速
  （vLLM 官方实测 1.5-3x）
- 落地点（待验证）：vllm `--speculative-config`（draft_model + num_speculative_tokens）
- 关键风险：RL 训练必须拿到目标分布精确 logprobs，vLLM 不保证 spec decode 下 logprob
  稳定性 → 需在 verl rollout 链路实测（distilled 类 draft 用 rejection sampling 保分布）
- 注意：draft 模型多占显存，适合双卡/多卡阶段；batch 大、KV 复用低时收益下降需实测阈值

## 4. PD 分离（Prefill/Decode Disaggregation，后续亮点）

- prefill 与 decode 拆不同 worker/实例，经 KV connector（MooncakeConnector）跨引擎传 KV，
  消除长 prompt prefill 对 decode 吞吐（TPOT）的抢占；vLLM 官方明确适配 agentic RL
- 与双机 TQ + Mooncake 结合：prefill 机与 decode 机分离部署，KV 传输走 Mooncake

## 5. 服务器恢复 checklist

1. 用户恢复 node2 镜像（新实例 ≥64GB 内存）→ 更新本地 `work/ucloud.env`（新公网/内网 IP）
2. node2：`git -C /home/ubuntu/uniagent-lighting pull`（本仓代码/文档/脚本）
3. 环境验证（import / 模型存在）→ 按 §1 构造 HumanEvalFix 数据 → 跑单机 agentic GRPO 冒烟
4. 双机网络就绪后：重做 hosts/SSH/Ray（`scripts/fix_multinode_hosts.sh` +
   `setup_ssh_trust.py` + `ray_cluster_setup.sh`）→ 开 TQ + Mooncake
