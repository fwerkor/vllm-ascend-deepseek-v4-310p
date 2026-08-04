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

    local_ids, local_mask = remap_global_expert_ids_310(topk_ids, expert_map)

    assert torch.equal(
        local_ids,
        torch.tensor([[0, 0, 3], [1, 0, 2]], dtype=torch.int32),
    )
    assert torch.equal(
        local_mask,
        torch.tensor([[False, True, True], [True, False, True]]),
    )
