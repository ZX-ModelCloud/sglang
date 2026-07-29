from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported
from sglang.srt.layers.quantization.utils import (
    get_dynamic_override,
    get_linear_quant_method,
    get_scalar_types,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

from .schemes import (
    GPTQAscendLinearScheme,
    GPTQIntelAMXLinearScheme,
    GPTQIntelAMXMoEScheme,
    GPTQLinearScheme,
    GPTQMarlinLinearScheme,
    GPTQMarlinMoEScheme,
    GPTQMoEAscendScheme,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)
_, scalar_types = get_scalar_types()


def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    # compat: gptqmodel and autogptq (eol) main use checkpoint_format: str
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )


@dataclass(frozen=True)
class _GPTQQuantConfigValues:
    weight_bits: int
    group_size: int
    desc_act: bool
    is_sym: bool


def _apply_dynamic_override(
    values: _GPTQQuantConfigValues,
    override: Optional[Dict[str, Union[int, bool]]],
) -> _GPTQQuantConfigValues:
    if override is None:
        return values
    return _GPTQQuantConfigValues(
        weight_bits=int(override.get("bits", values.weight_bits)),
        group_size=int(override.get("group_size", values.group_size)),
        desc_act=bool(override.get("desc_act", values.desc_act)),
        is_sym=bool(override.get("sym", values.is_sym)),
    )


def _clone_marlin_config(
    config: GPTQMarlinConfig,
    values: _GPTQQuantConfigValues,
) -> GPTQMarlinConfig:
    quant_type = config.TYPE_MAP.get((values.weight_bits, values.is_sym))
    if quant_type is None:
        raise ValueError(
            "Unsupported quantization config: "
            f"bits={values.weight_bits}, sym={values.is_sym}"
        )

    cloned_config = copy(config)
    cloned_config.full_config = config.full_config.copy()
    cloned_config.weight_bits = values.weight_bits
    cloned_config.group_size = values.group_size
    cloned_config.desc_act = values.desc_act
    cloned_config.is_sym = values.is_sym
    cloned_config.pack_factor = 32 // values.weight_bits
    cloned_config.quant_type = quant_type
    cloned_config.full_config.update(
        {
            "bits": values.weight_bits,
            "group_size": values.group_size,
            "desc_act": values.desc_act,
            "sym": values.is_sym,
        }
    )
    return cloned_config


def _resolve_moe_quant_config(
    config: GPTQMarlinConfig,
    layer: torch.nn.Module,
    prefix: str,
) -> tuple[Optional[GPTQMarlinConfig], Dict[tuple[int, str], int]]:
    """Resolve one losslessly representable GPTQ scheme for a fused MoE."""
    base_values = _GPTQQuantConfigValues(
        weight_bits=config.weight_bits,
        group_size=config.group_size,
        desc_act=config.desc_act,
        is_sym=config.is_sym,
    )

    layer_override = get_dynamic_override(config, layer_name=prefix)
    if layer_override is False:
        return None, {}
    assert layer_override is None or isinstance(layer_override, dict)
    layer_values = _apply_dynamic_override(base_values, layer_override)

    shard_projections = {
        "w1": getattr(layer, "ckpt_gate_proj_name", "gate_proj"),
        "w2": getattr(layer, "ckpt_down_proj_name", "down_proj"),
        "w3": getattr(layer, "ckpt_up_proj_name", "up_proj"),
    }
    num_checkpoint_experts = layer.num_global_routed_experts
    shard_values = {}
    for expert_id in range(num_checkpoint_experts):
        for shard_id, projection_name in shard_projections.items():
            shard_prefix = f"{prefix}.{expert_id}.{projection_name}"
            dynamic_override = get_dynamic_override(config, layer_name=shard_prefix)
            if dynamic_override is False:
                shard_values[(expert_id, shard_id)] = None
                continue
            assert dynamic_override is None or isinstance(dynamic_override, dict)
            shard_values[(expert_id, shard_id)] = _apply_dynamic_override(
                layer_values, dynamic_override
            )

    quantized_values = [value for value in shard_values.values() if value is not None]
    if not quantized_values:
        return None, {}
    if len(quantized_values) != len(shard_values):
        skipped_shards = [
            f"expert {expert_id} {shard_id}"
            for (expert_id, shard_id), value in shard_values.items()
            if value is None
        ]
        raise ValueError(
            f"GPTQ FusedMoE layer '{prefix}' excludes only some expert "
            f"shards from quantization: {skipped_shards}"
        )

    def get_uniform_value(name: str) -> Union[int, bool]:
        values = {getattr(value, name) for value in quantized_values}
        if len(values) != 1:
            raise ValueError(
                f"GPTQ FusedMoE layer '{prefix}' has mixed {name}: "
                f"{sorted(values)}. Only group_size may differ between experts."
            )
        return next(iter(values))

    weight_bits = int(get_uniform_value("weight_bits"))
    desc_act = bool(get_uniform_value("desc_act"))
    is_sym = bool(get_uniform_value("is_sym"))
    source_group_sizes = {
        shard: value.group_size
        for shard, value in shard_values.items()
        if value is not None
    }
    unique_group_sizes = set(source_group_sizes.values())

    if len(unique_group_sizes) == 1:
        group_size = next(iter(unique_group_sizes))
    else:
        if desc_act:
            raise ValueError(
                f"GPTQ FusedMoE layer '{prefix}' has mixed group_size "
                f"{sorted(unique_group_sizes)} with desc_act=True, which cannot "
                "be normalized losslessly."
            )
        if any(group_size <= 0 for group_size in unique_group_sizes):
            raise ValueError(
                f"GPTQ FusedMoE layer '{prefix}' cannot mix channelwise and "
                f"group quantization: {sorted(unique_group_sizes)}"
            )
        group_size = min(unique_group_sizes)
        if any(
            source_group_size % group_size for source_group_size in unique_group_sizes
        ):
            raise ValueError(
                f"GPTQ FusedMoE layer '{prefix}' has incompatible group "
                f"sizes: {sorted(unique_group_sizes)}"
            )
        logger.info_once(
            "GPTQ FusedMoE checkpoint uses mixed group sizes %s; "
            "normalizing quantization metadata to group_size=%d.",
            tuple(sorted(unique_group_sizes)),
            group_size,
        )

    resolved_values = _GPTQQuantConfigValues(
        weight_bits=weight_bits,
        group_size=group_size,
        desc_act=desc_act,
        is_sym=is_sym,
    )
    return _clone_marlin_config(config, resolved_values), source_group_sizes


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        checkpoint_format: str = "",
        true_sequential: bool = False,
        static_groups: bool = False,
        modules_in_block_to_quantize: Optional[List[Any]] = None,
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        # GPTQ v1 and v2 format deals with zero points differently.
        # Currently GPTQModel stores v1 format checkpoints by default,
        # but provides the option to set `format="gptq_v2"` in `QuantizeConfig`.
        self.checkpoint_format = checkpoint_format
        self.true_sequential = true_sequential
        self.static_groups = static_groups
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}),"
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic},"
            f"checkpoint_format={self.checkpoint_format})"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQConfig:
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        true_sequential = cls.get_from_keys_or(
            config, ["true_sequential"], default=False
        )
        static_groups = cls.get_from_keys_or(config, ["static_groups"], default=False)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        quant_config = cls(
            weight_bits,
            group_size,
            desc_act,
            lm_head_quantized,
            dynamic,
            checkpoint_format,
            true_sequential,
            static_groups,
            modules_in_block_to_quantize,
        )
        quant_config.update_packed_modules_mapping(
            config.get("packed_modules_mapping", {})
        )
        return quant_config

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            raise TypeError("GPTQ Method does not support MoE, please use gptq_marlin")
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        raise NotImplementedError("GPTQConfig does not support MoE.")


class GPTQAscendConfig(GPTQConfig):
    """Config class for GPTQ on Ascend NPU."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'NPU hardware does not support "get_min_capability" feature.'
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return GPTQMoEMethod(self)
        if isinstance(layer, LinearBase):
            layer.scheme = self.get_linear_scheme(layer)
            return GPTQLinearMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQAscendLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        return GPTQMoEAscendScheme(self)


class CPUGPTQConfig(GPTQConfig):
    """CPU Config class for GPTQ on Intel CPU with AMX."""

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            layer.scheme = self.get_linear_scheme(layer)
            return GPTQLinearMethod(self)
        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return GPTQMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.linear import LinearBase

        assert isinstance(layer, LinearBase)
        return GPTQIntelAMXLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        return GPTQIntelAMXMoEScheme(self)


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        full_config: Dict[str, Any],
        modules_in_block_to_quantize: Optional[List[Any]] = None,
    ) -> None:
        super().__init__()
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config
        self.modules_in_block_to_quantize = modules_in_block_to_quantize

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: " f"bits={weight_bits}, sym={is_sym}"
            )

        # (num_bits, is_sym) -> quant_type
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}), "
            f"dynamic={self.dynamic}"
        )

    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> GPTQMarlinConfig:
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        quant_config = cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            modules_in_block_to_quantize,
        )
        quant_config.update_packed_modules_mapping(
            config.get("packed_modules_mapping", {})
        )
        return quant_config

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)

        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )

        if not is_marlin_format and can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if not is_marlin_format and can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # Delay the import to avoid circular dependency
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            resolved_config, source_group_sizes = _resolve_moe_quant_config(
                self, layer, prefix
            )
            if resolved_config is None:
                return None
            has_mixed_group_sizes = len(set(source_group_sizes.values())) > 1
            return GPTQMarlinMoEMethod(
                resolved_config,
                source_group_sizes if has_mixed_group_sizes else None,
            )
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )

    def get_linear_scheme(self, layer: torch.nn.Module):
        return GPTQMarlinLinearScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        return GPTQMarlinMoEScheme(self)

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if quant_method != "gptq":
            return False

        # Marlin conversion is only valid if required properties are found
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        try:
            return check_marlin_supported(
                quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
            )
        except Exception:
            return False


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_linear_scheme(layer)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


class GPTQMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: GPTQConfig):
        super().__init__()
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_moe_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self,
        layer: torch.nn.Module,
        moe_runner_config: MoeRunnerConfig,
        **extra_weight_attrs,
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, dispatch_output)


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_linear_scheme(layer)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: GPTQMarlinConfig,
        source_group_sizes: Optional[Dict[tuple[int, str], int]] = None,
    ) -> None:
        self.quant_config = quant_config
        self.source_group_sizes = source_group_sizes or {}

    def _normalize_group_metadata(
        self,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> torch.Tensor:
        if not weight_name.endswith(("scales", "qzeros", "g_idx")):
            return loaded_weight

        source_group_size = self.source_group_sizes.get((expert_id, shard_id))
        if source_group_size is None:
            raise ValueError(
                "Missing source group_size for GPTQ FusedMoE "
                f"expert {expert_id} shard {shard_id}."
            )

        target_group_size = self.quant_config.group_size
        if source_group_size == target_group_size:
            return loaded_weight
        if (
            source_group_size <= 0
            or target_group_size <= 0
            or source_group_size % target_group_size
        ):
            raise ValueError(
                "Cannot normalize GPTQ FusedMoE group_size "
                f"{source_group_size} to {target_group_size} for expert "
                f"{expert_id} shard {shard_id}."
            )

        repeat_factor = source_group_size // target_group_size
        # Repeated metadata describes the same dequantized qweight values.
        if weight_name.endswith(("scales", "qzeros")):
            return loaded_weight.repeat_interleave(repeat_factor, dim=0)

        positions = torch.arange(
            loaded_weight.numel(),
            dtype=loaded_weight.dtype,
            device=loaded_weight.device,
        )
        expected_g_idx = (positions // source_group_size).reshape_as(loaded_weight)
        if not torch.equal(loaded_weight, expected_g_idx):
            raise ValueError(
                "Cannot normalize non-sequential g_idx for mixed-group "
                f"GPTQ FusedMoE expert {expert_id} shard {shard_id}."
            )
        return (positions // target_group_size).reshape_as(loaded_weight)

    def _wrap_weight_loader(self, weight_loader):
        def mixed_group_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ) -> None:
            loaded_weight = self._normalize_group_metadata(
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )

        return mixed_group_weight_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.source_group_sizes:
            extra_weight_attrs["weight_loader"] = self._wrap_weight_loader(
                extra_weight_attrs["weight_loader"]
            )
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_moe_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return layer.scheme.apply_weights(layer, dispatch_output)


# Register fake implementations for torch.compile support. The decorator is a
# no-op when the custom op is unavailable on the current platform.
@register_fake_if_exists("sgl_kernel::gptq_gemm")
def _(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit):
    return a.new_empty((a.shape[0], b_q_weight.shape[-1]), dtype=a.dtype)


@register_fake_if_exists("sgl_kernel::gptq_marlin_repack")
def _(b_q_weight, perm, size_k, size_n, num_bits):
    return b_q_weight.new_empty(
        (size_k // 16, size_n * (num_bits // 2)), dtype=b_q_weight.dtype
    )


@register_fake_if_exists("sgl_kernel::gptq_shuffle")
def _(q_weight, q_perm, bit):
    return
