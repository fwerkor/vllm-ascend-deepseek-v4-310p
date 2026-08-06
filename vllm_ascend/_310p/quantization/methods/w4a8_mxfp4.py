# SPDX-License-Identifier: Apache-2.0
"""DeepSeek MXFP4 checkpoint adapter for Ascend 310P W8A8 execution."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend._310p.quantization.methods.mxfp4_to_w8a8 import requantize_mxfp4_to_int8
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod310
from vllm_ascend.utils import maybe_trans_nz


def _is_preconverted_w8a8_layout(layer: torch.nn.Module) -> bool:
    """Return whether an expert layer already stores expanded W8A8 tensors.

    Raw DeepSeek MXFP4 and converted W8A8 both use byte-sized weight storage,
    so dtype alone cannot distinguish them.  The raw checkpoint keeps one
    E8M0 scale per 32 logical values (3-D scale tensors) and packs two FP4
    values per weight byte.  The converted checkpoint stores one FP32 scale
    per output row (2-D scale tensors) and full logical input widths.
    """
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    s13 = layer.w13_weight_scale
    s2 = layer.w2_weight_scale
    return (
        w13.dtype == torch.int8
        and w2.dtype == torch.int8
        and w13.ndim == 3
        and w2.ndim == 3
        and s13.dtype == torch.float32
        and s2.dtype == torch.float32
        and s13.ndim == 2
        and s2.ndim == 2
        and tuple(s13.shape) == tuple(w13.shape[:2])
        and tuple(s2.shape) == tuple(w2.shape[:2])
    )


class AscendMXFP4ToW8A8DynamicFusedMoEMethod310(AscendW8A8DynamicFusedMoEMethod310):
    """Load DeepSeek packed MXFP4 experts and execute through 310P W8A8 MoE.

    The checkpoint-facing parameter shapes remain packed MXFP4. After loading,
    each rank converts only its local expert shard to symmetric per-row INT8.
    """

    group_size = 32
    _MODE_ENV = "VLLM_ASCEND_DSV4_310P_EXPERT_MODE"
    _STREAMING_MODE = "streaming_w8a8"
    _EAGER_MODE = "eager_w8a8"
    _PRECONVERTED_MODE = "preconverted_w8a8"

    def __init__(self, quant_config: dict[str, Any], tid2eid=None):
        super().__init__()
        configured_group_size = quant_config.get("group_size", self.group_size)
        if configured_group_size != self.group_size:
            raise ValueError(
                f"DeepSeek V4 MXFP4 on 310P requires group_size={self.group_size}, got {configured_group_size}."
            )
        self.tid2eid = tid2eid
        self.execution_mode = os.getenv(self._MODE_ENV, self._STREAMING_MODE)
        if self.execution_mode not in (self._STREAMING_MODE, self._EAGER_MODE, self._PRECONVERTED_MODE):
            raise ValueError(
                f"Unsupported {self._MODE_ENV}={self.execution_mode!r}; "
                f"expected {self._STREAMING_MODE!r}, {self._EAGER_MODE!r}, "
                f"or {self._PRECONVERTED_MODE!r}."
            )

    @staticmethod
    def get_weight(
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // 2,
                dtype=torch.uint8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.float8_e8m0fnu,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e8m0fnu,
            ),
        }

    def process_weights_after_loading(self, layer) -> None:
        if _is_preconverted_w8a8_layout(layer):
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
            # Keep a canonical 2-D FP32 scale layout even if a loader supplied
            # an equivalent view.
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.to(torch.float32).view(
                layer.w13_weight_scale.shape[0], -1
            )
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.to(torch.float32).view(
                layer.w2_weight_scale.shape[0], -1
            )
            logger.info_once(
                "Loaded preconverted DeepSeek V4 W8A8 expert shards on Ascend 310P.",
                scope="local",
            )
            return

        if self.execution_mode == self._STREAMING_MODE:
            logger.info_once(
                "Keeping local DeepSeek V4 expert shards packed for 310P streaming W8A8: w13=%s, w2=%s, experts=%d.",
                tuple(layer.w13_weight.shape),
                tuple(layer.w2_weight.shape),
                layer.w13_weight.shape[0],
            )
            return

        if self.execution_mode == self._PRECONVERTED_MODE:
            logger.info_once(
                "Detected raw packed MXFP4 expert tensors while the target model uses preconverted W8A8; "
                "converting this draft expert shard eagerly instead of treating packed bytes as INT8 weights.",
                scope="local",
            )
        logger.info_once(
            "Eagerly converting local DeepSeek V4 MXFP4 expert shard to 310P W8A8: w13=%s, w2=%s, experts=%d.",
            tuple(layer.w13_weight.shape),
            tuple(layer.w2_weight.shape),
            layer.w13_weight.shape[0],
        )
        w13_weight, w13_scale = requantize_mxfp4_to_int8(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2_weight, w2_scale = requantize_mxfp4_to_int8(layer.w2_weight.data, layer.w2_weight_scale.data)

        layer.w13_weight.data = maybe_trans_nz(w13_weight)
        layer.w2_weight.data = maybe_trans_nz(w2_weight)
        layer.w13_weight_scale.data = w13_scale.view(w13_scale.shape[0], -1)
        layer.w2_weight_scale.data = w2_scale.view(w2_scale.shape[0], -1)
        logger.info_once(
            "Converted raw MXFP4 expert scales to W8A8: w13=[%.6g, %.6g], w2=[%.6g, %.6g].",
            float(layer.w13_weight_scale.min().item()),
            float(layer.w13_weight_scale.max().item()),
            float(layer.w2_weight_scale.min().item()),
            float(layer.w2_weight_scale.max().item()),
            scope="local",
        )

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        if self.execution_mode in (self._EAGER_MODE, self._PRECONVERTED_MODE):
            return super().apply(layer, *args, **kwargs)

        # Materialize one layer at a time. The NPU caching allocator reuses the
        # temporary buffers across decoder layers, keeping peak memory bounded
        # while the checkpoint remains in its compact packed representation.
        w13_weight, w13_scale = requantize_mxfp4_to_int8(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2_weight, w2_scale = requantize_mxfp4_to_int8(layer.w2_weight.data, layer.w2_weight_scale.data)

        materialized_layer = SimpleNamespace(
            w13_weight=maybe_trans_nz(w13_weight),
            w2_weight=maybe_trans_nz(w2_weight),
            w13_weight_scale=w13_scale.view(w13_scale.shape[0], -1),
            w2_weight_scale=w2_scale.view(w2_scale.shape[0], -1),
            zero_expert_num=getattr(layer, "zero_expert_num", 0),
            zero_expert_type=getattr(layer, "zero_expert_type", None),
        )
        return super().apply(materialized_layer, *args, **kwargs)
