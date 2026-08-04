#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections.abc import Callable

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.fused_moe.experts_selector import _native_select_experts, _renormalize_topk_weights


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    global_num_experts: int = -1,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """
    Fused experts with select experts.

    Args:
        router_logits: router logits of shape (num_tokens, hidden_size).
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        top_k: number of top k experts.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.
        routed_scaling_factor: Scaling factor applied to routing weights.
        global_num_experts: Global number of experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("Hash-routed MoE requires current input_ids.")
        token_ids = input_ids.reshape(-1).to(torch.int64)
        if token_ids.numel() != router_logits.shape[0]:
            raise ValueError(
                f"Hash-routed input_ids and router rows differ: {token_ids.numel()} vs {router_logits.shape[0]}."
            )
        token_ids = torch.where(token_ids < 0, torch.zeros_like(token_ids), token_ids)
        topk_ids = tid2eid.index_select(0, token_ids).to(torch.int32)
        if topk_ids.shape[-1] != top_k:
            raise ValueError(f"Hash table returns {topk_ids.shape[-1]} experts, expected top_k={top_k}.")

        # DeepSeek V4 hash layers use the token table only for expert IDs.
        # Routing weights still come from the unbiased router scores.
        if scoring_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            scores = router_logits.sigmoid()
        elif scoring_func == "sqrtsoftplus":
            scores = F.softplus(router_logits).sqrt()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")
        topk_weights = scores.gather(1, topk_ids.to(torch.int64))
        topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
        return topk_weights.to(hidden_states.dtype), topk_ids

    if scoring_func == "softmax" and not use_grouped_topk and custom_routing_function is None:
        # 310P returns invalid routing results when this op receives more than 1024 tokens.
        if router_logits.shape[0] > 1024:
            topk_results = [
                torch_npu.npu_moe_gating_top_k_softmax(router_logits_chunk, k=top_k)
                for router_logits_chunk in router_logits.split(1024, dim=0)
            ]
            topk_weights = torch.cat([result[0] for result in topk_results], dim=0)
            topk_ids = torch.cat([result[1] for result in topk_results], dim=0)
        else:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)
        topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    else:
        topk_weights, topk_ids = _native_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
    # Apply routed scaling factor to weights
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids
