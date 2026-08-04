# SPDX-License-Identifier: Apache-2.0
"""Correct short-context DeepSeek V4 attention fallbacks for Ascend 310P."""

from __future__ import annotations

import torch


def _flatten_rope_cache(cache: torch.Tensor, num_tokens: int, rotary_dim: int) -> torch.Tensor:
    """Normalize DeepSeek V4 RoPE cache to ``[T, rotary_dim]``."""
    return cache[:num_tokens].reshape(num_tokens, rotary_dim)


def apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply pairwise/interleaved RoPE to the trailing dimensions of ``x``.

    DeepSeek V4 stores each frequency twice, yielding cache rows like
    ``[cos0, cos0, cos1, cos1, ...]``.  Ascend 310P's ACLop rotary kernel only
    supports the half-rotation layout, so this function implements the exact
    interleaved formula with ordinary tensor operations.
    """
    if rotary_dim == 0:
        return x
    if rotary_dim % 2 != 0 or rotary_dim > x.shape[-1]:
        raise ValueError(
            f"rotary_dim must be even and no larger than the head dimension, got {rotary_dim} and {x.shape[-1]}"
        )

    num_tokens = x.shape[0]
    cos_flat = _flatten_rope_cache(cos, num_tokens, rotary_dim)
    sin_flat = _flatten_rope_cache(sin, num_tokens, rotary_dim)
    if inverse:
        sin_flat = -sin_flat

    prefix = x[..., :-rotary_dim]
    rotary = x[..., -rotary_dim:]
    rotary_shape = rotary.shape
    rotary_pairs = rotary.reshape(*rotary_shape[:-1], rotary_dim // 2, 2)

    # Frequency values are duplicated for the even and odd element in each
    # pair.  Shape [T, 1, rotary_dim/2] broadcasts over attention heads.
    cos_pairs = cos_flat[:, 0::2].unsqueeze(1).to(torch.float32)
    sin_pairs = sin_flat[:, 0::2].unsqueeze(1).to(torch.float32)
    even = rotary_pairs[..., 0].to(torch.float32)
    odd = rotary_pairs[..., 1].to(torch.float32)
    rotated = torch.stack(
        (
            even * cos_pairs - odd * sin_pairs,
            odd * cos_pairs + even * sin_pairs,
        ),
        dim=-1,
    ).reshape(rotary_shape)
    rotated = rotated.to(x.dtype)
    return torch.cat((prefix, rotated), dim=-1) if prefix.shape[-1] else rotated


def normalize_swa_cache(cache: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Unwrap the one-element cache containers used by compressed KV groups."""
    while isinstance(cache, (list, tuple)):
        if len(cache) != 1:
            raise ValueError(f"Expected one SWA cache tensor, got {len(cache)} entries")
        cache = cache[0]
    return cache


def write_paged_swa_cache(
    cache: torch.Tensor,
    kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write ``[T, 1, D]`` KV rows into a paged ``[B, S, 1, D]`` cache."""
    if slot_mapping is None:
        raise ValueError("SWA slot_mapping is required for the 310P dense fallback")
    if slot_mapping.ndim == 2:
        flat_slots = slot_mapping[:, 0].to(torch.int64) * block_size + slot_mapping[:, 1].to(torch.int64)
    elif slot_mapping.ndim == 1:
        flat_slots = slot_mapping.to(torch.int64)
    else:
        raise ValueError(f"Unsupported SWA slot_mapping shape: {tuple(slot_mapping.shape)}")

    valid = flat_slots >= 0
    if not bool(valid.any()):
        return
    flat_cache = cache.reshape(-1, cache.shape[-2], cache.shape[-1])
    flat_cache.index_copy_(0, flat_slots[valid], kv[valid].to(cache.dtype))


def infer_blocks_per_phys_block(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    input_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_size: int,
) -> int:
    """Infer the logical-to-physical block split used by hybrid block tables.

    vLLM may expose a cache group through a small logical kernel block while
    the underlying SWA tensor is allocated with a larger physical block.  In
    that case each physical block ``P`` appears in the block table as
    ``P * factor, ..., P * factor + factor - 1``.  ``slot_mapping`` remains in
    physical-cache coordinates, so it provides an exact runtime witness for
    recovering ``factor`` without depending on private block-table objects.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if slot_mapping is None:
        raise ValueError("SWA slot_mapping is required to infer hybrid block geometry")

    table_cpu = block_table.detach().to("cpu", dtype=torch.int64)
    if table_cpu.ndim == 1:
        table_cpu = table_cpu.unsqueeze(0)
    if table_cpu.ndim != 2:
        raise ValueError(f"Unsupported block_table shape: {tuple(block_table.shape)}")

    slots_cpu = slot_mapping.detach().to("cpu", dtype=torch.int64)
    if slots_cpu.ndim == 1:
        slot_blocks = torch.div(slots_cpu, block_size, rounding_mode="floor")
        slot_offsets = torch.remainder(slots_cpu, block_size)
    elif slots_cpu.ndim == 2 and slots_cpu.shape[1] == 2:
        slot_blocks = slots_cpu[:, 0]
        slot_offsets = slots_cpu[:, 1]
    else:
        raise ValueError(f"Unsupported SWA slot_mapping shape: {tuple(slot_mapping.shape)}")

    positions_cpu = input_positions.detach().to("cpu", dtype=torch.int64).reshape(-1)
    query_offsets = query_start_loc.detach().to("cpu", dtype=torch.int64).reshape(-1).tolist()
    num_tokens = min(positions_cpu.numel(), slot_blocks.numel())
    if not query_offsets or query_offsets[0] != 0 or query_offsets[-1] > num_tokens:
        raise ValueError(
            "query_start_loc is inconsistent with input_positions/slot_mapping: "
            f"offsets={query_offsets}, num_tokens={num_tokens}"
        )
    if len(query_offsets) - 1 > table_cpu.shape[0]:
        raise ValueError(f"block_table has {table_cpu.shape[0]} rows for {len(query_offsets) - 1} requests")

    candidates: list[int] = []
    for factor in range(1, block_size + 1):
        if block_size % factor != 0:
            continue
        logical_block_size = block_size // factor
        matched_valid_slot = False
        valid_candidate = True
        for request_idx in range(len(query_offsets) - 1):
            token_start = int(query_offsets[request_idx])
            token_end = min(int(query_offsets[request_idx + 1]), num_tokens)
            for token_idx in range(token_start, token_end):
                physical_block = int(slot_blocks[token_idx].item())
                physical_offset = int(slot_offsets[token_idx].item())
                if physical_block < 0 or physical_offset < 0:
                    continue
                matched_valid_slot = True
                position = int(positions_cpu[token_idx].item())
                logical_table_idx = position // logical_block_size
                if logical_table_idx >= table_cpu.shape[1]:
                    valid_candidate = False
                    break
                logical_block = int(table_cpu[request_idx, logical_table_idx].item())
                predicted_block = logical_block // factor
                predicted_offset = (logical_block % factor) * logical_block_size + position % logical_block_size
                if predicted_block != physical_block or predicted_offset != physical_offset:
                    valid_candidate = False
                    break
            if not valid_candidate:
                break
        if valid_candidate and matched_valid_slot:
            candidates.append(factor)

    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely infer hybrid block split from DSA metadata: "
            f"block_size={block_size}, candidates={candidates}"
        )
    return candidates[0]


def gather_paged_swa_cache(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    start: int,
    end: int,
    block_size: int,
    blocks_per_phys_block: int = 1,
) -> torch.Tensor:
    """Gather token range ``[start, end)`` from paged SWA storage.

    ``block_table_row`` may contain logical sub-block IDs produced by vLLM's
    hybrid block table.  Convert those IDs back to physical cache coordinates
    before indexing the physical ``[num_blocks, block_size, ...]`` tensor.
    """
    if end <= start:
        return cache.new_empty((0, cache.shape[-1]))
    if blocks_per_phys_block <= 0 or block_size % blocks_per_phys_block != 0:
        raise ValueError(
            "blocks_per_phys_block must be a positive divisor of block_size, "
            f"got {blocks_per_phys_block} and {block_size}"
        )
    logical_block_size = block_size // blocks_per_phys_block
    positions = torch.arange(start, end, device=cache.device, dtype=torch.int64)
    logical_table_indices = torch.div(positions, logical_block_size, rounding_mode="floor")
    table = block_table_row.to(device=cache.device, dtype=torch.int64)
    if int(logical_table_indices[-1].item()) >= table.numel():
        raise ValueError(
            f"Block table is too short for token range [{start}, {end}): "
            f"need index {int(logical_table_indices[-1].item())}, have {table.numel()} entries"
        )
    logical_blocks = table.index_select(0, logical_table_indices)
    physical_blocks = torch.div(logical_blocks, blocks_per_phys_block, rounding_mode="floor")
    offsets = torch.remainder(logical_blocks, blocks_per_phys_block) * logical_block_size + torch.remainder(
        positions, logical_block_size
    )
    if bool((physical_blocks < 0).any()) or bool((physical_blocks >= cache.shape[0]).any()):
        raise ValueError(
            "Hybrid block table resolved outside the SWA cache: "
            f"physical range=[{int(physical_blocks.min().item())}, "
            f"{int(physical_blocks.max().item())}], cache blocks={cache.shape[0]}"
        )
    return cache[physical_blocks, offsets, 0]


def dense_causal_current_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    window_size: int,
    softmax_scale: float,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Causal attention for a fresh prefill using only the current Q/KV.

    The current KV is also written to the paged cache by the caller for later
    decode steps. Avoiding an immediate cache round-trip prevents stale or
    incorrectly mapped physical blocks from affecting first-token logits.
    """
    output = torch.empty_like(q)
    query_offsets = query_start_loc.to("cpu").tolist()

    for request_idx in range(len(query_offsets) - 1):
        q_start = int(query_offsets[request_idx])
        q_end = int(query_offsets[request_idx + 1])
        request_kv = kv[q_start:q_end, 0]
        for local_query_idx in range(q_end - q_start):
            visible_end = local_query_idx + 1
            visible_start = max(0, visible_end - window_size)
            keys = request_kv[visible_start:visible_end]
            query = q[q_start + local_query_idx].to(keys.dtype)
            logits = torch.matmul(query, keys.transpose(0, 1)).to(torch.float32) * softmax_scale
            sink_logits = sinks.to(torch.float32).reshape(-1, 1)
            probabilities = torch.softmax(torch.cat((logits, sink_logits), dim=-1), dim=-1)[..., : keys.shape[0]]
            output[q_start + local_query_idx] = torch.matmul(probabilities.to(keys.dtype), keys).to(q.dtype)

    return output


def dense_causal_swa_attention(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    block_size: int,
    blocks_per_phys_block: int = 1,
    window_size: int,
    softmax_scale: float,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate causal sliding-window attention from the paged original KV.

    ``sinks`` are represented as one extra softmax logit per query head with a
    zero value vector, matching the DeepSeek sparse-attention kernel.
    """
    output = torch.empty_like(q)
    query_offsets = query_start_loc.to("cpu").tolist()
    sequence_lengths = seq_lens.to("cpu").tolist()

    for request_idx, seq_len_value in enumerate(sequence_lengths):
        q_start = int(query_offsets[request_idx])
        q_end = int(query_offsets[request_idx + 1])
        q_len = q_end - q_start
        seq_len = int(seq_len_value)
        context_len = seq_len - q_len
        if context_len < 0:
            raise ValueError(f"Sequence length {seq_len} is smaller than query length {q_len}")

        for local_query_idx in range(q_len):
            visible_end = context_len + local_query_idx + 1
            visible_start = max(0, visible_end - window_size)
            keys = gather_paged_swa_cache(
                cache,
                block_table[request_idx],
                visible_start,
                visible_end,
                block_size,
                blocks_per_phys_block,
            )
            query = q[q_start + local_query_idx].to(keys.dtype)
            logits = torch.matmul(query, keys.transpose(0, 1)).to(torch.float32) * softmax_scale
            sink_logits = sinks.to(torch.float32).reshape(-1, 1)
            probabilities = torch.softmax(torch.cat((logits, sink_logits), dim=-1), dim=-1)[..., : keys.shape[0]]
            output[q_start + local_query_idx] = torch.matmul(probabilities.to(keys.dtype), keys).to(q.dtype)

    return output
