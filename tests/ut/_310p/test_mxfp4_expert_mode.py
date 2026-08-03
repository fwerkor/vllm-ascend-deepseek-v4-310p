# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch


def test_streaming_is_the_default_mode() -> None:
    from vllm_ascend._310p.quantization.methods.w4a8_mxfp4 import (
        AscendMXFP4ToW8A8DynamicFusedMoEMethod310,
    )

    method = AscendMXFP4ToW8A8DynamicFusedMoEMethod310.__new__(
        AscendMXFP4ToW8A8DynamicFusedMoEMethod310
    )
    with patch.dict("os.environ", {}, clear=True):
        mode = __import__("os").getenv(method._MODE_ENV, method._STREAMING_MODE)
    assert mode == method._STREAMING_MODE


def test_supported_expert_modes_are_explicit() -> None:
    from vllm_ascend._310p.quantization.methods.w4a8_mxfp4 import (
        AscendMXFP4ToW8A8DynamicFusedMoEMethod310,
    )

    assert {
        AscendMXFP4ToW8A8DynamicFusedMoEMethod310._STREAMING_MODE,
        AscendMXFP4ToW8A8DynamicFusedMoEMethod310._EAGER_MODE,
    } == {"streaming_w8a8", "eager_w8a8"}
