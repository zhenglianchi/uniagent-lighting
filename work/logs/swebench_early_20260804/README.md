# 早期单样本轨迹归档（2026-08-04 前后）

## 说明

链路打通初期的单样本轨迹与日志（SWE-bench + HumanEvalFix），用于学习
`*.traj.json` 的字段结构（agent 动作、工具调用、奖励）。

- `tencent_django_13447.traj.json`：SWE-bench django 单样本（腾讯沙箱）
- `tencent_marshmallow-code__marshmallow-1359.traj.json`：marshmallow 单样本
- `sympy__sympy-13043*.traj.json`：sympy 单样本（含 step40 截断版）
- `humanevalfix_humanevalfix-Python-*.traj.json`：HumanEvalFix 早期样本
- `*_run.log` / `humanevalfix_local_summary.json`：运行日志与本地摘要
