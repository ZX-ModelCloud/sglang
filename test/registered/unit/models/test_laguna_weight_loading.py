"""CPU-only tests for Laguna checkpoint name mapping."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.laguna import LagunaForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLagunaWeightLoading(unittest.TestCase):
    def test_plural_shared_experts_checkpoint_prefix_is_loaded(self):
        model = LagunaForCausalLM.__new__(LagunaForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=0,
            mlp_layer_types=["dense"],
            tie_word_embeddings=False,
        )

        model.model = nn.Module()
        model.model.start_layer = 0
        model.model.end_layer = 1
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.layers[0].mlp = nn.Module()
        model.model.layers[0].mlp.shared_experts = nn.Module()
        model.model.layers[0].mlp.shared_experts.gate_up_proj = nn.Module()

        loaded = {}

        def weight_loader(param, loaded_weight, shard_id):
            loaded["weight"] = loaded_weight
            loaded["shard_id"] = shard_id

        param = nn.Parameter(torch.empty(2, 2), requires_grad=False)
        param.weight_loader = weight_loader
        model.model.layers[0].mlp.shared_experts.gate_up_proj.register_parameter(
            "qweight", param
        )

        checkpoint_weight = torch.ones(2, 2)
        model.load_weights(
            [
                (
                    "model.layers.0.mlp.shared_experts.gate_proj.qweight",
                    checkpoint_weight,
                )
            ]
        )

        self.assertIs(loaded["weight"], checkpoint_weight)
        self.assertEqual(loaded["shard_id"], 0)


if __name__ == "__main__":
    unittest.main()
