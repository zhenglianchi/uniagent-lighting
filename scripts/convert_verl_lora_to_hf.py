#!/usr/bin/env python3
"""verl FSDP2 + LoRA checkpoint -> 合并后的标准 HF 模型目录（供 vLLM 直接 serve）。

verl 保存的 ``model_world_size_1_rank_0.pt`` 是 PEFT 风格 state dict：
  ``base_model.model.model.layers.N.self_attn.q_proj.base_layer.weight``
  ``base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight``
  ``base_model.model.model.layers.N.self_attn.q_proj.lora_B.default.weight``

本脚本把 LoRA 合并进 base（scale = alpha / r），输出标准 HF 模型
（config + tokenizer 从基座模型目录复制，权重转 safetensors）。

用法：
  python scripts/convert_verl_lora_to_hf.py \\
      --ckpt /home/ubuntu/swe-rl/checkpoints/humanevalfix/final/actor/model_world_size_1_rank_0.pt \\
      --base /home/ubuntu/models/Qwen3-8B \\
      --out /home/ubuntu/models/Qwen3-8B-final
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def load_state_dict(ckpt_path: str) -> dict:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def merge_lora(
    sd: dict,
    *,
    r: int,
    alpha: int,
) -> dict:
    """把 PEFT 风格 state dict 合并成标准 HF 权重（去掉 base_model 前缀和 lora 参数）。"""
    merged: dict[str, torch.Tensor] = {}
    scale = alpha / r
    lora_modules: dict[str, tuple[str, str, str]] = {}  # module -> (base_key, lora_A_key, lora_B_key)

    for key in sd:
        if key.endswith(".base_layer.weight"):
            module = key[: -len(".base_layer.weight")]
            lora_modules.setdefault(module, [key, None, None])[0] = key
        elif key.endswith(".lora_A.default.weight"):
            module = key[: -len(".lora_A.default.weight")]
            lora_modules.setdefault(module, [None, key, None])[1] = key
        elif key.endswith(".lora_B.default.weight"):
            module = key[: -len(".lora_B.default.weight")]
            lora_modules.setdefault(module, [None, None, key])[2] = key

    done: set[str] = set()
    for module, (base_key, lora_a_key, lora_b_key) in lora_modules.items():
        assert base_key and lora_a_key and lora_b_key, f"incomplete lora module: {module}"
        base = sd[base_key].float()
        lora_a = sd[lora_a_key].float()
        lora_b = sd[lora_b_key].float()
        delta = (lora_b @ lora_a) * scale
        out_key = "base_model.model." + module + ".weight"
        merged[out_key] = (base + delta).to(base.dtype)
        done.update({base_key, lora_a_key, lora_b_key})

    # 非 LoRA 模块直接透传（embed / norm / lm_head 等）
    for key, tensor in sd.items():
        if key in done:
            continue
        if key.startswith("base_model.model."):
            merged[key] = tensor
        else:
            print(f"[skip] non-base key: {key}")
    return merged


def save_hf_model(merged: dict, out_dir: Path, base_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 标准权重名：去掉 "base_model.model." 前缀
    renamed = {k[len("base_model.model.") :]: v for k, v in merged.items()}
    try:
        from safetensors.torch import save_file

        meta = {"format": "pt"}
        save_file(renamed, str(out_dir / "model.safetensors"), metadata=meta)
    except ImportError:
        torch.save(renamed, str(out_dir / "pytorch_model.bin"))
        print("[warn] safetensors 不可用，落盘 pytorch_model.bin")
    # config + tokenizer 从基座目录复制
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    print(f"[ok] merged HF model saved to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="verl checkpoint 的 model_world_size_1_rank_0.pt")
    p.add_argument("--base", required=True, help="基座模型目录（config/tokenizer 来源）")
    p.add_argument("--out", required=True, help="输出合并模型目录")
    p.add_argument("--lora-meta", default="", help="lora_train_meta.json 路径（r/alpha；缺省默认 32/32）")
    args = p.parse_args()

    r, alpha = 32, 32
    if args.lora_meta:
        meta = json.loads(Path(args.lora_meta).read_text())
        r = int(meta.get("r", r))
        alpha = int(meta.get("lora_alpha", alpha))
    print(f"[info] r={r} alpha={alpha}")

    sd = load_state_dict(args.ckpt)
    merged = merge_lora(sd, r=r, alpha=alpha)
    save_hf_model(merged, Path(args.out), Path(args.base))


if __name__ == "__main__":
    main()
