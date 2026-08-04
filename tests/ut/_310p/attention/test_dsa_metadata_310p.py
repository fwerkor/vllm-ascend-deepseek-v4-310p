# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.ops.dsa import filter_exact_metadata


def test_filter_metadata_selects_exact_swa_namespace() -> None:
    expected = object()
    metadata = {
        "model.layers.2.self_attn.attn": object(),
        "model.layers.2.self_attn.compressor.state_cache": object(),
        "model.layers.2.self_attn.indexer.k_cache": object(),
        "model.layers.2.self_attn.swa_cache": expected,
    }

    assert filter_exact_metadata(metadata, "model.layers.2.self_attn.swa_cache") == [expected]


def test_filter_metadata_rejects_missing_or_ambiguous_namespace() -> None:
    with pytest.raises(ValueError, match="Expected exactly one"):
        filter_exact_metadata({}, "model.layers.2.self_attn.swa_cache")

    metadata = {
        "model.layers.2.self_attn.swa_cache": object(),
        "model.layers.2.self_attn.swa_cache.extra": object(),
    }
    with pytest.raises(ValueError, match="got 2"):
        filter_exact_metadata(metadata, "model.layers.2.self_attn.swa_cache")
