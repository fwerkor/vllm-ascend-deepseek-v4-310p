# SPDX-License-Identifier: Apache-2.0

import os

from vllm_ascend._310p import determinism


def test_dsv4_determinism_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSV4_310P_DETERMINISTIC", raising=False)
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    calls = []
    monkeypatch.setattr(determinism.torch_npu.npu, "set_deterministic_level", calls.append)

    determinism.configure_dsv4_determinism()

    assert calls == []
    assert "HCCL_DETERMINISTIC" not in os.environ


def test_dsv4_determinism_configures_hccl_and_aclnn(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSV4_310P_DETERMINISTIC", "1")
    monkeypatch.delenv("HCCL_DETERMINISTIC", raising=False)
    calls = []
    monkeypatch.setattr(determinism.torch_npu.npu, "set_deterministic_level", calls.append)

    determinism.configure_dsv4_determinism()

    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    assert calls == [1]
