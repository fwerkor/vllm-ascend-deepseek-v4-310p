# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend._310p.fused_moe import token_dispatcher as dispatcher_module
from vllm_ascend._310p.fused_moe.token_dispatcher import (
    remap_global_expert_ids_310,
    resolve_live_ep_rank_310,
)


def test_resolve_live_ep_rank_310_ignores_stale_cached_rank(monkeypatch):
    ep_group = SimpleNamespace(
        rank_in_group=0,
        ranks=[0, 1, 2, 3, 4, 5, 6, 7],
        device_group=object(),
    )
    monkeypatch.setattr(dispatcher_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 6)

    assert resolve_live_ep_rank_310() == 6


def test_remap_global_expert_ids_310_masks_non_local_routes():
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[4:8] = torch.arange(4, dtype=torch.int32)
    topk_ids = torch.tensor([[0, 4, 7], [5, 2, 6]], dtype=torch.int32)

    local_ids, local_mask = remap_global_expert_ids_310(topk_ids, expert_map, non_local_expert_id=4)

    assert torch.equal(
        local_ids,
        torch.tensor([[4, 0, 3], [1, 4, 2]], dtype=torch.int32),
    )
    assert torch.equal(
        local_mask,
        torch.tensor([[False, True, True], [True, False, True]]),
    )


def test_dispatch_uses_trailing_virtual_expert_to_exclude_non_local_rows(monkeypatch):
    dispatcher = dispatcher_module.TokenDispatcherWithAllGather310(
        top_k=3,
        num_experts=8,
        num_local_experts=4,
    )
    hidden_states = torch.randn(2, 32)
    topk_ids = torch.tensor([[0, 4, 7], [5, 2, 6]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[4:8] = torch.arange(4, dtype=torch.int32)

    calls = {}

    def fake_init_routing(x, expert_ids, **kwargs):
        calls["x"] = x
        calls["expert_ids"] = expert_ids
        calls["kwargs"] = kwargs
        sorted_hidden_states = torch.randn(6, 32)
        expanded_row_idx = torch.tensor([4, 0, 3, 1, 5, 2], dtype=torch.int32)
        # Four local routes followed by two virtual-expert routes.
        expert_tokens = torch.tensor([1, 2, 3, 4, 6], dtype=torch.int32)
        return sorted_hidden_states, expanded_row_idx, expert_tokens, None

    monkeypatch.setattr(
        dispatcher_module.torch_npu,
        "npu_moe_init_routing_v2",
        fake_init_routing,
        raising=False,
    )
    output = dispatcher.token_dispatch(
        SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            routing=SimpleNamespace(
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            ),
        )
    )

    assert torch.equal(
        calls["expert_ids"],
        torch.tensor([[4, 0, 3], [1, 4, 2]], dtype=torch.int32),
    )
    assert calls["kwargs"]["expert_num"] == 5
    assert calls["kwargs"]["active_expert_range"] == [0, 5]
    assert torch.equal(output.group_list, torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    assert torch.equal(
        output.combine_metadata.topk_weights,
        torch.tensor([[0.0, 0.3, 0.5], [0.1, 0.0, 0.5]]),
    )
