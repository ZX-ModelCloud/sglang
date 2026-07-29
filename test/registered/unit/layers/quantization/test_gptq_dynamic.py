"""CPU-only tests for dynamic GPTQ fused-layer configuration."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.gptq.gptq import (
    GPTQMarlinConfig,
    GPTQMarlinMoEMethod,
    _resolve_moe_quant_config,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import (
    GPTQMarlinMoEScheme,
)
from sglang.srt.layers.quantization.utils import (
    get_dynamic_override,
    is_layer_gptq_quantized,
)
from sglang.srt.model_loader.weight_utils import _get_gptq_quantized_modules
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MOE_PREFIX = "model.layers.0.mlp.experts"


def _exact_rule(module_name: str, **override):
    return f"+:^{re.escape(module_name)}$", override


def _make_config(
    dynamic,
    *,
    group_size=128,
    desc_act=False,
    packed_modules_mapping=None,
    modules_in_block_to_quantize=None,
):
    full_config = {
        "bits": 4,
        "group_size": group_size,
        "desc_act": desc_act,
        "sym": True,
        "lm_head": False,
        "quant_method": "gptq",
        "dynamic": dynamic,
        "packed_modules_mapping": packed_modules_mapping
        or {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        },
        "modules_in_block_to_quantize": modules_in_block_to_quantize,
    }
    return GPTQMarlinConfig.from_config(full_config)


def _make_moe_stub(num_experts=3):
    return SimpleNamespace(
        num_global_routed_experts=num_experts,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )


class TestGPTQDynamic(unittest.TestCase):
    def test_moe_scales_follow_activation_dtype(self):
        scheme = GPTQMarlinMoEScheme(_make_config({}, group_size=32))
        layer = torch.nn.Module()
        layer.moe_tp_size = 1

        scheme.create_weights(
            layer,
            num_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.w13_scales.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_scales.dtype, torch.bfloat16)

    def test_checkpoint_modules_override_stale_dynamic_rules(self):
        weight_map = {
            "model.layers.0.self_attn.q_proj.qweight": "model.safetensors",
            "model.layers.0.self_attn.k_proj.qweight": "model.safetensors",
            "model.layers.0.self_attn.v_proj.qweight": "model.safetensors",
            "model.layers.0.self_attn.g_proj.weight": "model.safetensors",
        }
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
            quantized_modules = _get_gptq_quantized_modules(model_dir)

        assert quantized_modules is not None
        config = _make_config(
            dict(
                [
                    _exact_rule(
                        "model.layers.0.self_attn.g_proj",
                        bits=4,
                        group_size=32,
                    )
                ]
            ),
            modules_in_block_to_quantize=quantized_modules,
        )
        self.assertEqual(config.modules_in_block_to_quantize, quantized_modules)
        self.assertTrue(
            is_layer_gptq_quantized(
                "model.layers.0.self_attn.qkv_proj",
                quantized_modules,
                config.packed_modules_mapping,
            )
        )
        self.assertFalse(
            is_layer_gptq_quantized(
                "model.layers.0.self_attn.g_proj",
                quantized_modules,
                config.packed_modules_mapping,
            )
        )

    def test_fused_qkv_uses_logical_shard_overrides(self):
        dynamic = dict(
            _exact_rule(
                f"model.layers.0.self_attn.{projection}",
                bits=4,
                group_size=32,
            )
            for projection in ("q_proj", "k_proj", "v_proj")
        )
        config = _make_config(dynamic)

        self.assertEqual(
            get_dynamic_override(
                config,
                "model.layers.0.self_attn.qkv_proj",
                "group_size",
                config.group_size,
            ),
            32,
        )
        self.assertEqual(
            config.packed_modules_mapping["qkv_proj"],
            ["q_proj", "k_proj", "v_proj"],
        )

    def test_fused_match_respects_module_boundaries(self):
        dynamic = dict(
            [
                _exact_rule(
                    f"model.layers.0.self_attn.{projection}",
                    group_size=32,
                )
                for projection in ("q_proj", "k_proj", "v_proj")
            ]
        )
        config = _make_config(dynamic)

        self.assertEqual(
            get_dynamic_override(
                config,
                "model.layers.0.self_attn.notqkv_proj",
                "group_size",
                config.group_size,
            ),
            128,
        )

    def test_fused_layer_rejects_inconsistent_shards(self):
        config = _make_config(
            dict(
                [
                    _exact_rule(
                        "model.layers.0.self_attn.q_proj",
                        group_size=32,
                    )
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "does not match across shards"):
            get_dynamic_override(
                config,
                "model.layers.0.self_attn.qkv_proj",
                "group_size",
                config.group_size,
            )

    def test_fused_layer_exclusion_must_cover_all_shards(self):
        config = _make_config({r"-:^model\.layers\.0\.self_attn\.q_proj$": {}})

        with self.assertRaisesRegex(ValueError, "does not match across shards"):
            get_dynamic_override(config, "model.layers.0.self_attn.qkv_proj")

        config.dynamic = {
            rf"-:^model\.layers\.0\.self_attn\.{projection}$": {}
            for projection in ("q_proj", "k_proj", "v_proj")
        }
        self.assertIs(
            get_dynamic_override(config, "model.layers.0.self_attn.qkv_proj"),
            False,
        )

    def test_dynamic_rules_keep_first_match_order(self):
        config = _make_config(
            {
                r"+:^model\.layers\..*\.q_proj$": {"group_size": 64},
                r"+:^model\.layers\.0\.q_proj$": {"group_size": 32},
            }
        )
        self.assertEqual(
            get_dynamic_override(
                config,
                "model.layers.0.q_proj",
                "group_size",
                config.group_size,
            ),
            64,
        )

    def test_moe_resolves_compatible_mixed_group_sizes(self):
        dynamic = {
            rf"+:^{re.escape(_MOE_PREFIX)}\.1\..*_proj$": {
                "bits": 4,
                "group_size": 32,
            },
            rf"+:^{re.escape(_MOE_PREFIX)}\.2\.down_proj$": {
                "bits": 4,
                "group_size": 32,
            },
        }
        config = _make_config(dynamic)

        resolved, source_group_sizes = _resolve_moe_quant_config(
            config,
            _make_moe_stub(),
            _MOE_PREFIX,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.group_size, 32)
        self.assertEqual(resolved.full_config["group_size"], 32)
        self.assertEqual(config.full_config["group_size"], 128)
        self.assertEqual(source_group_sizes[(0, "w1")], 128)
        self.assertEqual(source_group_sizes[(1, "w1")], 32)
        self.assertEqual(source_group_sizes[(2, "w1")], 128)
        self.assertEqual(source_group_sizes[(2, "w2")], 32)

    def test_moe_normalizes_group_metadata(self):
        method = GPTQMarlinMoEMethod(
            _make_config({}, group_size=32),
            {(0, "w1"): 128},
        )

        scales = torch.arange(6, dtype=torch.float16).reshape(2, 3)
        normalized_scales = method._normalize_group_metadata(
            scales,
            "model.layers.0.mlp.experts.0.gate_proj.scales",
            "w1",
            0,
        )
        self.assertTrue(
            torch.equal(normalized_scales, scales.repeat_interleave(4, dim=0))
        )

        qzeros = torch.arange(4, dtype=torch.int32).reshape(2, 2)
        normalized_qzeros = method._normalize_group_metadata(
            qzeros,
            "model.layers.0.mlp.experts.0.gate_proj.qzeros",
            "w1",
            0,
        )
        self.assertTrue(
            torch.equal(normalized_qzeros, qzeros.repeat_interleave(4, dim=0))
        )

        g_idx = torch.arange(256, dtype=torch.int32) // 128
        normalized_g_idx = method._normalize_group_metadata(
            g_idx,
            "model.layers.0.mlp.experts.0.gate_proj.g_idx",
            "w1",
            0,
        )
        self.assertTrue(
            torch.equal(
                normalized_g_idx,
                torch.arange(256, dtype=torch.int32) // 32,
            )
        )

    def test_moe_weight_loader_normalizes_before_loading(self):
        method = GPTQMarlinMoEMethod(
            _make_config({}, group_size=32),
            {(0, "w1"): 128},
        )
        loaded = {}

        def weight_loader(param, weight, weight_name, shard_id, expert_id):
            loaded[weight_name] = weight

        wrapped_loader = method._wrap_weight_loader(weight_loader)
        scales = torch.arange(6, dtype=torch.float16).reshape(2, 3)
        wrapped_loader(
            torch.nn.Parameter(torch.empty(0), requires_grad=False),
            scales,
            "model.layers.0.mlp.experts.0.gate_proj.scales",
            "w1",
            0,
        )

        self.assertTrue(
            torch.equal(
                loaded["model.layers.0.mlp.experts.0.gate_proj.scales"],
                scales.repeat_interleave(4, dim=0),
            )
        )

    def test_moe_rejects_non_sequential_group_indices(self):
        method = GPTQMarlinMoEMethod(
            _make_config({}, group_size=32),
            {(0, "w1"): 128},
        )

        with self.assertRaisesRegex(ValueError, "non-sequential g_idx"):
            method._normalize_group_metadata(
                torch.zeros(256, dtype=torch.int32),
                "model.layers.0.mlp.experts.0.gate_proj.g_idx",
                "w1",
                0,
            )

    def test_moe_rejects_mixed_groups_with_desc_act(self):
        dynamic = {
            rf"+:^{re.escape(_MOE_PREFIX)}\.1\..*_proj$": {
                "group_size": 32,
            }
        }

        with self.assertRaisesRegex(ValueError, "desc_act=True"):
            _resolve_moe_quant_config(
                _make_config(dynamic, desc_act=True),
                _make_moe_stub(num_experts=2),
                _MOE_PREFIX,
            )

    def test_moe_rejects_partial_exclusion(self):
        dynamic = {
            rf"-:^{re.escape(_MOE_PREFIX)}\.1\.down_proj$": {},
        }

        with self.assertRaisesRegex(ValueError, "excludes only some expert shards"):
            _resolve_moe_quant_config(
                _make_config(dynamic),
                _make_moe_stub(num_experts=2),
                _MOE_PREFIX,
            )

    def test_moe_rejects_incompatible_group_sizes(self):
        dynamic = {
            rf"+:^{re.escape(_MOE_PREFIX)}\.1\..*_proj$": {
                "group_size": 96,
            }
        }

        with self.assertRaisesRegex(ValueError, "incompatible group sizes"):
            _resolve_moe_quant_config(
                _make_config(dynamic),
                _make_moe_stub(num_experts=2),
                _MOE_PREFIX,
            )


if __name__ == "__main__":
    unittest.main()
