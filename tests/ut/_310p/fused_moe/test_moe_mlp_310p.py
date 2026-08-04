# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.fused_moe.moe_mlp import (
    clipped_swiglu_310p,
    zero_inactive_grouped_matmul_rows,
)


def test_clipped_swiglu_matches_deepseek_v4_formula():
    gate_up = torch.tensor(
        [[12.0, -12.0, 3.0, -4.0, 15.0, -15.0, 2.0, -3.0]],
        dtype=torch.float32,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(up, min=-10.0, max=10.0)
    actual = clipped_swiglu_310p(gate_up, limit=10.0)
    torch.testing.assert_close(actual, expected)


def test_clipped_swiglu_uses_fp32_and_stays_bounded() -> None:
    from vllm_ascend._310p.fused_moe.moe_mlp import clipped_swiglu_310p

    gate_up = torch.tensor([[512.0, -512.0, 512.0, -512.0]], dtype=torch.float16)
    actual = clipped_swiglu_310p(gate_up, limit=10.0)
    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(up, min=-10.0, max=10.0)).to(
        torch.float16
    )
    torch.testing.assert_close(actual, expected)
    assert actual.abs().max().item() <= 100.0


def test_zero_inactive_grouped_matmul_rows_clears_nan_tail() -> None:
    hidden_states = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [float("nan"), 9.0],
            [-7.0, float("nan")],
        ]
    )
    cumulative_group_list = torch.tensor([1, 2, 2], dtype=torch.int64)

    actual = zero_inactive_grouped_matmul_rows(
        hidden_states,
        cumulative_group_list,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_zero_inactive_grouped_matmul_rows_handles_no_active_rows() -> None:
    hidden_states = torch.full((3, 2), float("nan"))
    cumulative_group_list = torch.zeros(4, dtype=torch.int64)

    actual = zero_inactive_grouped_matmul_rows(
        hidden_states,
        cumulative_group_list,
    )

    torch.testing.assert_close(actual, torch.zeros_like(hidden_states))
