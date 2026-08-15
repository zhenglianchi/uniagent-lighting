#!/usr/bin/env python3
"""应用 PR #7014 修复：merge 分支改为 context 内物化的生成器。

背景：get_per_tensor_param() 的 merge 分支在 merged_lora_context 内提取
state_dict()，但返回的是生成器，consumer（update_weights）在 context 退出
（基座权重已恢复）后才迭代物化 → 同步给 vLLM 的是没有 LoRA 的基座权重，
策略更新从未真正作用到 rollout（2026-08-09 训练 reward 不升的根因）。
"""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("/home/ubuntu/uni-agent/verl/verl/workers/engine/fsdp/transformer_impl.py")

OLD_MERGE = """            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
"""

NEW_MERGE = """            else:  # merge lora
                # state_dict() aliases the live parameter storage and merged_lora_context
                # restores the un-merged base weights on exit, so tensors must be
                # materialized while the context is still open (inside the generator).
                # Materializing after exit silently sends base weights without adapters.
                return self._merged_lora_per_tensor_param(), None
"""

NEW_METHOD = '''
    def _merged_lora_per_tensor_param(self):
        """Stream merged (base + LoRA) weights for rollout weight sync.

        ``state_dict()`` returns tensors that alias the live FSDP parameter
        storage, and ``merged_lora_context`` restores the un-merged base
        weights when it exits. The context therefore must stay open until the
        consumer has materialized every tensor: ``DTensor.full_tensor()``
        produces a copy, so yielded tensors remain valid after the restore.
        Consuming a state_dict captured inside the context after the context
        has exited would silently send base weights without the adapters.
        """
        device = get_device_id()
        try:
            with merged_lora_context(self.module, backup_adapters=True):
                params = normalize_peft_param_name(self.module.state_dict())
                params = convert_weight_keys(
                    params, getattr(self.module, "_fsdp_wrapped_module", self.module)
                )
                for name, param in params.items():
                    yield (
                        name,
                        param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                        if isinstance(param, DTensor)
                        # clone: plain tensors also alias module storage, and bucketed
                        # senders may flush after the restore has already run
                        else param.detach().clone(),
                    )
        finally:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

    def disable_adapter(self) -> ContextManager:
'''


def main() -> None:
    src = PATH.read_text(encoding="utf-8")
    if OLD_MERGE not in src:
        raise SystemExit("OLD_MERGE block not found — patch may already be applied or file changed")
    src = src.replace(OLD_MERGE, NEW_MERGE, 1)

    old_anchor = "    def disable_adapter(self) -> ContextManager:\n"
    if old_anchor not in src:
        raise SystemExit("disable_adapter anchor not found")
    if "def _merged_lora_per_tensor_param" in src:
        raise SystemExit("method already exists")
    src = src.replace(old_anchor, NEW_METHOD, 1)

    PATH.write_text(src, encoding="utf-8")
    print("patched OK:", PATH)


if __name__ == "__main__":
    main()
