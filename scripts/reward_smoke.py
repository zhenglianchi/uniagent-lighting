"""冒烟奖励函数：回答非空给 1 分（用于验证训练链路，非真实 SWE-bench 奖励）。"""


def smoke_reward(data_source, solution_str, ground_truth, extra_info=None):
    """非空回答即 +1，用于冒烟验证。"""
    if not solution_str or not solution_str.strip():
        return 0.0
    return 1.0


# verl 自定义 reward 约定函数名：compute_score
compute_score = smoke_reward
