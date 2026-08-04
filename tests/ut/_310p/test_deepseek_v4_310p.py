# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend._310p.deepseek_v4 import (
    DSA_BACKEND_310P,
    DSV4_310P_ENV,
    get_dsv4_310p_backend,
    is_deepseek_v4_model,
    is_dsv4_310p_enabled,
)


def test_dsv4_310p_is_opt_in() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert not is_dsv4_310p_enabled()
        assert get_dsv4_310p_backend(use_mla=True, use_sparse=False, use_compress=True) is None


def test_dsv4_310p_selects_dsa_for_compressed_mla() -> None:
    with patch.dict("os.environ", {DSV4_310P_ENV: "1"}, clear=True):
        assert get_dsv4_310p_backend(use_mla=True, use_sparse=False, use_compress=True) == DSA_BACKEND_310P


def test_dsv4_310p_selects_dsa_without_compression() -> None:
    with patch.dict("os.environ", {DSV4_310P_ENV: "1"}, clear=True):
        assert get_dsv4_310p_backend(use_mla=True, use_sparse=False, use_compress=False) == DSA_BACKEND_310P


def test_dsv4_310p_does_not_override_sfa_or_dense_attention() -> None:
    with patch.dict("os.environ", {DSV4_310P_ENV: "1"}, clear=True):
        assert get_dsv4_310p_backend(use_mla=True, use_sparse=True, use_compress=False) is None
        assert get_dsv4_310p_backend(use_mla=False, use_sparse=False, use_compress=False) is None


def test_deepseek_v4_model_detection_is_version_tolerant() -> None:
    dsv4 = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4"))
    qwen = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3"))
    assert is_deepseek_v4_model(dsv4)
    assert not is_deepseek_v4_model(qwen)
    assert not is_deepseek_v4_model(SimpleNamespace())


def test_w8a8_moe_marks_routed_output_as_scaled() -> None:
    from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
        AscendW8A8DynamicFusedMoEMethod310,
    )

    assert AscendW8A8DynamicFusedMoEMethod310.routed_output_includes_scale is True


def test_deepseek_v4_scaled_routed_output_is_added_without_rescaling() -> None:
    import inspect

    from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE

    source = inspect.getsource(DeepseekV4MoE.forward)
    assert "routed_output_includes_scale" in source
    assert "final_hidden_states = shared_output + final_hidden_states" in source


def test_310p_lm_head_bypasses_global_fp8_config() -> None:
    import inspect

    from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM

    source = inspect.getsource(AscendDeepseekV4ForCausalLM.__init__)
    assert "quant_config=None if is_310p() else quant_config" in source


def test_deepseek_v4_passes_swiglu_limit_to_routed_experts() -> None:
    import inspect

    from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE

    source = inspect.getsource(DeepseekV4MoE.__init__)
    assert "swiglu_limit=self.swiglu_limit" in source
