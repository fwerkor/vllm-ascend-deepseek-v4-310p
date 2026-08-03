# SPDX-License-Identifier: Apache-2.0
"""Feature gates and backend selection for experimental DeepSeek V4 on 310P."""

from __future__ import annotations

import os

DSV4_310P_ENV = "VLLM_ASCEND_ENABLE_DSV4_310P"

MLA_BACKEND_310P = "vllm_ascend._310p.attention.mla_v1.AscendMLABackend310"
DSA_BACKEND_310P = "vllm_ascend._310p.attention.dsa_v1.AscendDSABackend310"


def is_dsv4_310p_enabled() -> bool:
    """Return whether the experimental 310P DeepSeek V4 path is enabled."""
    return os.getenv(DSV4_310P_ENV, "0") == "1"


def get_dsv4_310p_backend(
    *,
    use_mla: bool,
    use_sparse: bool,
    use_compress: bool,
) -> str | None:
    """Return a dedicated 310P backend path for MLA/DSA configurations.

    DeepSeek V4 uses the compressed DSA path represented by
    ``use_mla=True, use_sparse=False, use_compress=True``. Plain MLA is kept as
    a separate extension point because both paths need different metadata and
    KV-cache handling.
    """
    if not is_dsv4_310p_enabled() or not use_mla or use_sparse:
        return None
    return DSA_BACKEND_310P if use_compress else MLA_BACKEND_310P


def is_deepseek_v4_model(model_config) -> bool:
    """Identify DeepSeek V4 without depending on a specific vLLM config API."""
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_text_config, "model_type", None) == "deepseek_v4"
