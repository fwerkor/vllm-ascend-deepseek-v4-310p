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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from vllm.config.load import LoadConfig

from vllm_ascend._310p.sharded_state_loader_310p import (
    DSV4_W8A8_LOAD_FORMAT,
    DSV4_W8A8_MARKER,
    ShardedStateLoader310,
)


class TinyModel(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...] = (2, 3), dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        self.register_buffer("counter", torch.zeros(1, dtype=torch.int32))


def _write_marker(path: Path, tensor_parallel_size: int = 1) -> None:
    (path / DSV4_W8A8_MARKER).write_text(
        json.dumps(
            {
                "format": DSV4_W8A8_LOAD_FORMAT,
                "version": 1,
                "tensor_parallel_size": tensor_parallel_size,
            }
        ),
        encoding="utf-8",
    )


def _model_config(path: Path):
    return SimpleNamespace(model=str(path), model_weights=None)


def test_save_model_streams_multiple_safetensor_parts(tmp_path: Path) -> None:
    model = TinyModel(shape=(4, 4), dtype=torch.float32)
    model.weight.data.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))

    with patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0):
        ShardedStateLoader310.save_model(model, str(tmp_path), max_size=32)

    parts = sorted(tmp_path.glob("model-rank-0-part-*.safetensors"))
    assert len(parts) == 2
    merged = {}
    for part in parts:
        merged.update(load_file(str(part)))
    torch.testing.assert_close(merged["weight"], model.weight.cpu())
    torch.testing.assert_close(merged["counter"], model.counter.cpu())


def test_load_weights_replaces_parameter_shape_and_dtype(tmp_path: Path) -> None:
    _write_marker(tmp_path)
    checkpoint_weight = torch.arange(12, dtype=torch.int8).reshape(3, 4)
    save_file(
        {"weight": checkpoint_weight, "counter": torch.tensor([7], dtype=torch.int32)},
        str(tmp_path / "model-rank-0-part-0.safetensors"),
    )
    model = TinyModel(shape=(2, 3), dtype=torch.float32)
    loader = ShardedStateLoader310(LoadConfig(load_format=DSV4_W8A8_LOAD_FORMAT))

    with (
        patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        loader.load_weights(model, _model_config(tmp_path))

    assert model.weight.shape == (3, 4)
    assert model.weight.dtype == torch.int8
    torch.testing.assert_close(model.weight.cpu(), checkpoint_weight)
    assert model.counter.item() == 7


def test_load_weights_skips_target_keys_unused_by_draft(tmp_path: Path) -> None:
    _write_marker(tmp_path)
    checkpoint_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file(
        {
            "weight": checkpoint_weight,
            "counter": torch.tensor([2], dtype=torch.int32),
            "target_only.weight": torch.ones(8, dtype=torch.float32),
        },
        str(tmp_path / "model-rank-0-part-0.safetensors"),
    )
    model = TinyModel()
    loader = ShardedStateLoader310(LoadConfig(load_format=DSV4_W8A8_LOAD_FORMAT))

    with (
        patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        loader.load_weights(model, _model_config(tmp_path))

    torch.testing.assert_close(model.weight.cpu(), checkpoint_weight)
    assert model.counter.item() == 2


def test_load_weights_rejects_tensor_parallel_mismatch(tmp_path: Path) -> None:
    _write_marker(tmp_path, tensor_parallel_size=8)
    save_file(
        {"weight": torch.zeros((2, 3)), "counter": torch.zeros(1, dtype=torch.int32)},
        str(tmp_path / "model-rank-0-part-0.safetensors"),
    )
    loader = ShardedStateLoader310(LoadConfig(load_format=DSV4_W8A8_LOAD_FORMAT))

    with (
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=4),
        pytest.raises(ValueError, match="tensor-parallel size mismatch"),
    ):
        loader.load_weights(TinyModel(), _model_config(tmp_path))


def test_generate_quant_description_and_marker(tmp_path: Path) -> None:
    model = TinyModel(shape=(2, 3), dtype=torch.int8)

    with (
        patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=8),
    ):
        ShardedStateLoader310.generate_quant_description(model, str(tmp_path))

    quant_description = json.loads((tmp_path / "parameters_type_map.json").read_text(encoding="utf-8"))
    marker = json.loads((tmp_path / DSV4_W8A8_MARKER).read_text(encoding="utf-8"))
    assert quant_description["model_quant_type"] == "W8A8_DYNAMIC"
    assert quant_description["weight"] == "W8A8_DYNAMIC"
    assert quant_description["counter"] == "W8A8_DYNAMIC"
    assert marker["format"] == DSV4_W8A8_LOAD_FORMAT
    assert marker["tensor_parallel_size"] == 8
