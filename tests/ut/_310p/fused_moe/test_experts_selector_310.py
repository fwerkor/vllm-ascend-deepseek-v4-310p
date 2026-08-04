#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import patch

import pytest
import torch

from vllm_ascend._310p.fused_moe.experts_selector import select_experts


class TestExpertsSelector310:
    @pytest.mark.parametrize("global_num_experts", [256, 128])
    def test_select_experts(self, global_num_experts):
        hidden_states = torch.randn(8, 16)
        router_logits = torch.randn(8, 8)

        with patch("torch_npu.npu_moe_gating_top_k_softmax") as mock_npu:
            mock_npu.return_value = (
                torch.randn(8, 2),
                torch.randint(0, 8, (8, 2), dtype=torch.int32),
                None,
            )

            topk_weights, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=2,
                use_grouped_topk=False,
                renormalize=True,
                topk_group=None,
                num_expert_group=None,
                custom_routing_function=None,
                scoring_func="softmax",
                e_score_correction_bias=None,
                global_num_experts=global_num_experts,
            )

            mock_npu.assert_called_once()

        assert topk_weights.shape == (8, 2)
        assert topk_ids.shape == (8, 2)

    def test_select_experts_chunks_large_token_batch(self):
        num_tokens = 2050
        hidden_states = torch.randn(num_tokens, 16)
        router_logits = torch.randn(num_tokens, 8)

        def mock_gating(logits, k):
            return (
                torch.ones(logits.shape[0], k),
                torch.zeros(logits.shape[0], k, dtype=torch.int32),
                None,
            )

        with patch(
            "torch_npu.npu_moe_gating_top_k_softmax",
            side_effect=mock_gating,
        ) as mock_npu:
            topk_weights, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=2,
                use_grouped_topk=False,
                renormalize=True,
                custom_routing_function=None,
                scoring_func="softmax",
            )

        assert [call.args[0].shape[0] for call in mock_npu.call_args_list] == [1024, 1024, 2]
        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)
        assert torch.all(topk_weights == 0.5)

    def test_hash_routing_uses_tid2eid_and_router_weights(self):
        hidden_states = torch.zeros(3, 4, dtype=torch.float16)
        router_logits = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0, 2.0],
            ],
            dtype=torch.float32,
        )
        tid2eid = torch.tensor(
            [
                [3, 1],
                [0, 2],
                [1, 3],
                [2, 0],
            ],
            dtype=torch.int32,
        )
        input_ids = torch.tensor([0, 2, 3], dtype=torch.int64)

        weights, ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            input_ids=input_ids,
            tid2eid=tid2eid,
        )

        expected_ids = tid2eid[input_ids]
        scores = torch.nn.functional.softplus(router_logits).sqrt()
        expected_weights = scores.gather(1, expected_ids.long())
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        expected_weights = expected_weights * 1.5

        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(weights.float(), expected_weights, rtol=2e-3, atol=2e-3)


def test_w8a8_apply_uses_instance_hash_table():
    import inspect

    from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
        AscendW8A8DynamicFusedMoEMethod310,
    )

    source = inspect.getsource(AscendW8A8DynamicFusedMoEMethod310.apply)
    assert 'getattr(self, "tid2eid", None)' in source
    assert "tid2eid=routing_table" in source
