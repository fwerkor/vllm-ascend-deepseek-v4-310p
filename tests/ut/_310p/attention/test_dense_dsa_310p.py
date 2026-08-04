# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.attention.dense_dsa import (
    dense_causal_current_attention,
    gather_paged_swa_cache,
    infer_blocks_per_phys_block,
)


def test_fresh_prefill_attention_matches_manual_causal_reference() -> None:
    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, -1.0]],
            [[-0.5, 1.0], [0.25, 0.75]],
        ],
        dtype=torch.float32,
    )
    kv = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    sinks = torch.tensor([0.1, -0.2], dtype=torch.float32)

    actual = dense_causal_current_attention(q, kv, offsets, window_size=2, softmax_scale=0.5, sinks=sinks)

    expected = torch.empty_like(q)
    keys_all = kv[:, 0]
    for i in range(3):
        start = max(0, i + 1 - 2)
        keys = keys_all[start : i + 1]
        logits = q[i] @ keys.T * 0.5
        probs = torch.softmax(torch.cat((logits, sinks[:, None]), dim=-1), dim=-1)[:, : keys.shape[0]]
        expected[i] = probs @ keys

    torch.testing.assert_close(actual, expected)


def test_infers_hybrid_block_split_from_slot_mapping() -> None:
    # Physical block 2 is exposed as four logical blocks [8, 9, 10, 11].
    block_table = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)
    positions = torch.arange(6, dtype=torch.int64)
    slot_mapping = torch.stack(
        (
            torch.full((6,), 2, dtype=torch.int32),
            torch.arange(6, dtype=torch.int32),
        ),
        dim=-1,
    )

    factor = infer_blocks_per_phys_block(
        block_table,
        slot_mapping,
        positions,
        torch.tensor([0, 6], dtype=torch.int32),
        block_size=8,
    )

    assert factor == 4


def test_gather_paged_cache_decodes_hybrid_logical_blocks() -> None:
    cache = torch.empty((4, 8, 1, 1), dtype=torch.float32)
    for block in range(cache.shape[0]):
        for offset in range(cache.shape[1]):
            cache[block, offset, 0, 0] = block * 100 + offset

    # Physical blocks 2 and 3, each split into four logical blocks of size 2.
    block_table_row = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32)
    actual = gather_paged_swa_cache(
        cache,
        block_table_row,
        start=0,
        end=10,
        block_size=8,
        blocks_per_phys_block=4,
    )

    expected = torch.tensor(
        [[200], [201], [202], [203], [204], [205], [206], [207], [300], [301]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_gather_paged_cache_preserves_unsplit_layout() -> None:
    cache = torch.arange(3 * 4, dtype=torch.float32).view(3, 4, 1, 1)
    actual = gather_paged_swa_cache(
        cache,
        torch.tensor([1, 2], dtype=torch.int32),
        start=0,
        end=6,
        block_size=4,
    )

    expected = torch.tensor([[4], [5], [6], [7], [8], [9]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
