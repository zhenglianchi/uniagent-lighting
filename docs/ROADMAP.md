# Roadmap（2026-08-11 更新；进度按 TODO.md / CHANGELOG.md 为准）

> 训练主力 = node1（117.50.189.37，1×4090 48G / 94G，2026-08-08 新建）；
> 双机目标 = node1 + node2（2×4090 24G），VPC 未通；所有代码/文档由本仓维护，
> 服务器 `git -C /home/ubuntu/uniagent-lighting pull` 即可续跑。

## 1. 换数据集：HumanEvalFix（✅ 已完成，2026-08-08~10 全样本跑通）

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
- ✅ 阶段一验收（2026-08-08，117.50.81.187→189.37）：P61 4/4=1.0、P104 1/4=1.0，
  **advantage≠0（max 1.5 / min -0.5）**——Qwen3-8B 在 HumanEvalFix 上有可训练奖励；
  中途修复提示词（heredoc 整文件重写约束，v0.28.3）
- ✅ 全样本 baseline（2026-08-08~09）：train161 / 26 步 = 5 epoch + 1 步，
  基座 76.4% → **final 83.2%（+6.8pp）**；评测与逐步统计见 `docs/训练评测分析.md`

## 1b. 黑盒采样（Claude Code，✅ 小样本验证通过 2026-08-12，正式训练已启动）

- runner：`uni_agent_ext/agents/claude_code_runner.py`（腾讯 E2B direct-URL 版）——
  `ANTHROPIC_BASE_URL` 直连公网 Gateway（去 `/v1`）+ 沙箱内 npm 装 pin 版
  claude-code（2.1.153，< 2.1.154）+ reward 复用 SWE-bench 评估
- 本地工具就绪：ccglass 1.1.2 + claude-code 2.1.153（npmmirror）
- ✅ 小样本：train3 × n=4 = 12/12 会话 reward 1.0（3 步 GRPO + LoRA 热插 +
  checkpoint），轨迹/排障见 `work/logs/blackbox_smoke_20260812/`
- ✅ 正式训练已启动：`run_grpo_humanevalfix_blackbox_ucloud.sh`
  （train161 / batch32 / mini16 / micro4 / 并发 64 / max_turns 60，baseline 同口径）
- 细节见 TODO §G

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

- **✅ 已完成（2026-08-09~10）**：EAGLE-3（`RedHatAI/Qwen3-8B-speculator.eagle3`，
  vLLM 0.11.1 V1 支持；独立 draft 小模型方案已因 V1 移除不可用）+
  `lora.merge=True`（LoRA×SD 互斥）；修 logprobs 0 全丢（0→1）与 merge 权重物化
  bug（backport verl#7014）
- **实测**：rollout 吞吐 **+41.7%**（199.2 → 282.4 tok/s），每 token 延迟 -39.5%，
  25 步全程稳定；最终评测 **82.61%（133/161）** vs baseline 83.2%（几乎持平）
- 细节见 TODO §9.1/§9.3 与 `docs/训练评测分析.md`

## 4. 服务器恢复 checklist

1. node1（117.50.189.37）已就绪（2026-08-08 镜像恢复，代码 v0.35.x）；node2 恢复镜像后
   更新本地 `work/ucloud.env`（新公网/内网 IP）
2. 服务器：`git -C /home/ubuntu/uniagent-lighting pull`（本仓代码/文档/脚本）
3. 环境验证（import / 模型存在）→ 单机黑盒 GRPO 冒烟（§1b）
4. 双机网络就绪后（同 VPC/子网）：重做 hosts/SSH/Ray（`scripts/fix_multinode_hosts.sh`
   + `nccl_multinode_test.py`）→ 跑双机全异步 GRPO（§2）
