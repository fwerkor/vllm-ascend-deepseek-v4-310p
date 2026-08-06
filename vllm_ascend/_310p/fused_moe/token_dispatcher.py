# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEAllGatherCombineMetadata, MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithAllGather


def resolve_live_ep_rank_310() -> int:
    """Resolve the worker-local EP rank after rfork.

    The inherited ``rank_in_group`` field can remain zero in every worker.
    Derive the local rank from the current distributed process and the EP
    group's global-rank ordering instead.
    """
    ep_group = get_ep_group()
    global_rank = int(torch.distributed.get_rank())
    try:
        return tuple(int(rank) for rank in ep_group.ranks).index(global_rank)
    except (AttributeError, ValueError):
        return int(torch.distributed.get_rank(group=ep_group.device_group))


def remap_global_expert_ids_310(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    non_local_expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global expert IDs to local slots for the 310P routing operator.

    The CANN 310P ``npu_moe_init_routing_v2`` implementation requires
    ``expert_num`` to be the local expert count and ignores
    ``active_expert_range`` when global IDs are supplied.  Non-local routes are
    mapped to a trailing virtual expert.  Routing therefore keeps the fixed
    expanded shape, while the grouped-matmul boundary can exclude the virtual
    expert rows entirely. Their combine weights are also masked to zero.
    """
    mapped = expert_map[topk_ids]
    local_mask = mapped >= 0
    local_ids = torch.where(local_mask, mapped, torch.full_like(mapped, non_local_expert_id))
    return local_ids.to(topk_ids.dtype), local_mask


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        restore_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            routing_topk_ids, mask = remap_global_expert_ids_310(
                topk_ids,
                expert_map,
                self.num_experts_local,
            )
            topk_weights = topk_weights * mask
            routing_expert_num = self.num_experts_local + 1
        else:
            routing_topk_ids = topk_ids
            routing_expert_num = self.num_experts_local

        assert hidden_states.shape[-1] % 16 == 0, (
            f"The last dim of hidden_states {hidden_states.shape[-1]} should be aligned with 16."
        )
        sorted_hidden_states, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            routing_topk_ids,
            active_num=num_tokens * self.top_k,
            expert_num=routing_expert_num,
            drop_pad_mode=0,
            active_expert_range=[0, routing_expert_num],
            quant_mode=-1,
            row_idx_type=0,
        )
        if expert_map is not None:
            # The virtual expert is sorted after all physical local experts.
            # Excluding its final cumulative boundary makes grouped matmul
            # process only local routes; the existing inactive-row clearing
            # keeps the fixed tail safe for token unpermutation.
            expert_tokens = expert_tokens[: self.num_experts_local]
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 0  # `cumsum` mode

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )
