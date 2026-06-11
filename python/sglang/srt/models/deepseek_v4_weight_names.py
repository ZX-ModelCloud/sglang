from __future__ import annotations

from typing import Optional, Tuple


def get_fused_wqkv_a_weight_info(
    name: str,
) -> Optional[Tuple[str, str, Optional[int]]]:
    """Return (param_name, shard_key, cat_dim) for fused wqkv_a loading.

    `cat_dim=None` means both source tensors are equivalent for the fused
    parameter and only one of them needs to be loaded.
    """
    for source_name, shard_key in ((".wq_a.", "q"), (".wkv.", "kv")):
        if source_name not in name:
            continue

        param_name = name.replace(source_name, ".wqkv_a.")
        suffix_to_cat_dim = {
            ".weight": 0,
            ".weight_scale": 0,
            ".weight_scale_inv": 0,
            ".qweight": 1,
            ".qzeros": 1,
            ".scales": 1,
            ".g_idx": None,
        }
        for suffix, cat_dim in suffix_to_cat_dim.items():
            if name.endswith(suffix):
                return param_name, shard_key, cat_dim

    return None


def remap_transformers_weight_name_to_sglang_format(name: str) -> str:
    """Map transformers DeepSeek-V4 weight keys to sglang module names."""
    indexer_name_mapping = {
        ".self_attn.compressor.indexer.q_b_proj.": ".self_attn.indexer.wq_b.",
        ".self_attn.compressor.indexer.weights_proj.": ".self_attn.indexer.weights_proj.",
        ".self_attn.compressor.indexer.kv_proj.": ".self_attn.indexer.compressor.wkv.",
        ".self_attn.compressor.indexer.gate_proj.": ".self_attn.indexer.compressor.wgate.",
        ".self_attn.compressor.indexer.kv_norm.": ".self_attn.indexer.compressor.norm.",
        ".self_attn.compressor.indexer.position_bias": ".self_attn.indexer.compressor.ape",
    }
    for transformers_name, sglang_name in indexer_name_mapping.items():
        name = name.replace(transformers_name, sglang_name)

    attn_name_mapping = {
        ".self_attn.q_a_proj.": ".self_attn.wq_a.",
        ".self_attn.q_b_proj.": ".self_attn.wq_b.",
        ".self_attn.kv_proj.": ".self_attn.wkv.",
        ".self_attn.o_a_proj.": ".self_attn.wo_a.",
        ".self_attn.o_b_proj.": ".self_attn.wo_b.",
        ".self_attn.q_a_norm.": ".self_attn.q_norm.",
    }
    for transformers_name, sglang_name in attn_name_mapping.items():
        name = name.replace(transformers_name, sglang_name)
    name = name.replace(".self_attn.sinks", ".self_attn.attn_sink")

    compressor_name_mapping = {
        ".compressor.kv_proj.": ".compressor.wkv.",
        ".compressor.gate_proj.": ".compressor.wgate.",
        ".compressor.kv_norm.": ".compressor.norm.",
    }
    for transformers_name, sglang_name in compressor_name_mapping.items():
        name = name.replace(transformers_name, sglang_name)
    name = name.replace(".compressor.position_bias", ".compressor.ape")

    mhc_name_mapping = {
        ".attn_hc.fn": ".hc_attn_fn",
        ".attn_hc.base": ".hc_attn_base",
        ".attn_hc.scale": ".hc_attn_scale",
        ".ffn_hc.fn": ".hc_ffn_fn",
        ".ffn_hc.base": ".hc_ffn_base",
        ".ffn_hc.scale": ".hc_ffn_scale",
        "model.hc_head.hc_fn": "model.hc_head_fn",
        "model.hc_head.hc_base": "model.hc_head_base",
        "model.hc_head.hc_scale": "model.hc_head_scale",
    }
    for transformers_name, sglang_name in mhc_name_mapping.items():
        name = name.replace(transformers_name, sglang_name)

    return name
