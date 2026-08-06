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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import ShardedStateLoader

logger = init_logger(__name__)

DSV4_W8A8_LOAD_FORMAT = "dsv4_310p_w8a8"
DSV4_W8A8_MARKER = "dsv4_310p_w8a8.json"
DSV4_W8A8_HEADER_ONLY_ENV = "VLLM_ASCEND_DSV4_310P_HEADER_ONLY"
_DEFAULT_MAX_FILE_SIZE = 4 * 1024**3
_LOADER_REGISTERED = False

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


class ShardedStateLoader310(ShardedStateLoader):
    """Sharded-state persistence for preconverted DeepSeek V4 310P W8A8.

    The source checkpoint constructs FP8 and packed-MXFP4 Parameters. The
    persisted checkpoint contains their post-load INT8 replacements, whose
    shapes and dtypes differ. The stock sharded loader can only copy into the
    original storage, so this loader replaces Parameter storage when needed.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    @staticmethod
    def _model_weights_path(model_config: ModelConfig) -> str:
        if model_config.model_weights:
            return model_config.model_weights
        return model_config.model

    @staticmethod
    def _load_marker(path: str) -> dict[str, Any]:
        marker_path = Path(path) / DSV4_W8A8_MARKER
        if not marker_path.is_file():
            raise ValueError(
                f"{DSV4_W8A8_LOAD_FORMAT} requires {marker_path}; "
                "the directory is not a preconverted DeepSeek V4 310P checkpoint."
            )
        with marker_path.open(encoding="utf-8") as f:
            marker = json.load(f)
        if marker.get("format") != DSV4_W8A8_LOAD_FORMAT:
            raise ValueError(f"Invalid DeepSeek V4 310P checkpoint marker: {marker!r}")
        return marker

    @staticmethod
    def _replace_buffer(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
        module_name, _, buffer_name = key.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        if buffer_name not in module._buffers:
            raise KeyError(f"Could not resolve buffer {key!r} while loading preconverted checkpoint.")
        module._buffers[buffer_name] = tensor

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        from safetensors.torch import safe_open
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        local_model_path = self._model_weights_path(model_config)
        marker = self._load_marker(local_model_path)
        expected_tp = marker.get("tensor_parallel_size")
        actual_tp = get_tensor_model_parallel_world_size()
        if expected_tp is not None and int(expected_tp) != actual_tp:
            raise ValueError(
                "Preconverted DeepSeek V4 checkpoint tensor-parallel size mismatch: "
                f"saved={expected_tp}, requested={actual_tp}."
            )

        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(local_model_path, self.pattern.format(rank=rank, part="*"))
        filepaths = sorted(glob.glob(pattern))
        if not filepaths:
            raise ValueError(f"Could not find checkpoint files {pattern!r}.")

        # Keep duplicate state-dict entries. They are written independently on
        # disk, avoiding shared-storage and non-contiguous/NZ serialization
        # problems and preserving tied parameters through ordinary copies.
        expected_keys = set(model.state_dict().keys())
        parameters = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))
        loaded_keys: set[str] = set()
        skipped_keys = 0
        start_time = time.perf_counter()
        header_only = os.getenv(DSV4_W8A8_HEADER_ONLY_ENV, "0") == "1"

        for filepath in filepaths:
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key not in expected_keys:
                        # MTP draft models use a subset of the target state.
                        # Skip without materializing the unrelated tensor.
                        skipped_keys += 1
                        continue
                    if header_only:
                        tensor_slice = f.get_slice(key)
                        shape = tuple(tensor_slice.get_shape())
                        dtype_name = str(tensor_slice.get_dtype())
                        dtype = _SAFETENSORS_DTYPES.get(dtype_name)
                        if dtype is None:
                            raise TypeError(
                                f"Unsupported safetensors dtype {dtype_name!r} for header-only tensor {key!r}."
                            )
                        cpu_tensor = None
                    else:
                        cpu_tensor = f.get_tensor(key).contiguous()
                        shape = tuple(cpu_tensor.shape)
                        dtype = cpu_tensor.dtype
                    if key in parameters:
                        target = parameters[key]
                        if tuple(target.shape) == shape and target.dtype == dtype:
                            # copy_ performs the host-to-device transfer directly;
                            # avoid a second full NPU tensor for unchanged weights.
                            if cpu_tensor is not None:
                                target.data.copy_(cpu_tensor)
                        else:
                            target.data = (
                                torch.empty(shape, dtype=dtype, device=target.device)
                                if cpu_tensor is None
                                else cpu_tensor.to(device=target.device)
                            )
                    elif key in buffers:
                        target = buffers[key]
                        if tuple(target.shape) == shape and target.dtype == dtype:
                            if cpu_tensor is not None:
                                target.copy_(cpu_tensor)
                        else:
                            replacement = (
                                torch.empty(shape, dtype=dtype, device=target.device)
                                if cpu_tensor is None
                                else cpu_tensor.to(device=target.device)
                            )
                            self._replace_buffer(model, key, replacement)
                    else:
                        raise KeyError(f"Could not resolve checkpoint tensor {key!r} in model state.")
                    loaded_keys.add(key)

        missing = expected_keys - loaded_keys
        if missing:
            sample = tuple(sorted(missing)[:32])
            raise ValueError(f"Missing {len(missing)} keys in preconverted checkpoint; first keys: {sample}")

        logger.info_once(
            "Loaded %d preconverted DeepSeek V4 W8A8 tensors for TP rank %d in %.2f seconds; "
            "skipped %d tensors not used by this model view.",
            len(loaded_keys),
            rank,
            time.perf_counter() - start_time,
            skipped_keys,
            scope="local",
        )
        if header_only:
            logger.warning_once(
                "Loaded DeepSeek V4 preconverted target in header-only diagnostic mode; "
                "weights are intentionally uninitialized and must not be used for inference.",
                scope="local",
            )

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        """Stream each rank's state to CPU-backed safetensor parts.

        Copying one tensor at a time avoids retaining a second full 36+ GiB
        rank state in host memory. Persisting logical ND tensors also strips
        device-specific FRACTAL_NZ descriptors; process_weights_after_loading
        reconstructs those descriptors on reload.
        """
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = pattern or ShardedStateLoader.DEFAULT_PATTERN
        max_size = _DEFAULT_MAX_FILE_SIZE if max_size is None else max_size
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}.")

        rank = get_tensor_model_parallel_rank()
        old_pattern = pattern.format(rank=rank, part="*")
        for old_file in output_dir.glob(old_pattern):
            old_file.unlink()

        part_idx = 0
        part_size = 0
        part: dict[str, torch.Tensor] = {}

        def flush() -> None:
            nonlocal part_idx, part_size, part
            if not part:
                return
            filename = pattern.format(rank=rank, part=part_idx)
            destination = output_dir / filename
            temporary = output_dir / f".{filename}.tmp-{os.getpid()}"
            save_file(part, str(temporary), metadata={"format": "pt"})
            os.replace(temporary, destination)
            logger.info(
                "Saved DeepSeek V4 310P rank %d part %d with %d tensors (%.2f GiB).",
                rank,
                part_idx,
                len(part),
                part_size / 1024**3,
            )
            part_idx += 1
            part_size = 0
            part = {}

        # Do not call ShardedStateLoader._filter_subtensors here. Converted
        # linear weights are transposed NZ views, and the generic filter calls
        # view(-1), which is invalid for those tensors. CPU-contiguous copies
        # no longer share storage, so saving every state-dict key is safe.
        for key, tensor in model.state_dict().items():
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
            tensor_size = cpu_tensor.numel() * cpu_tensor.element_size()
            if part and part_size + tensor_size > max_size:
                flush()
            part[key] = cpu_tensor
            part_size += tensor_size
        flush()

    @staticmethod
    def generate_quant_description(
        model: torch.nn.Module,
        path: str,
        quant_config=None,
    ) -> None:
        """Write rank-independent checkpoint metadata from TP rank zero."""
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_rank() != 0:
            return
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        quant_description: dict[str, str] = {
            "model_quant_type": "W8A8_DYNAMIC",
            "version": "1.0.0",
        }
        for name, tensor in model.state_dict().items():
            quant_description[name] = (
                "W8A8_DYNAMIC" if tensor.dtype in (torch.int8, torch.int32, torch.int64) else "FLOAT"
            )

        json_path = output_dir / "parameters_type_map.json"
        temporary = json_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(quant_description, f, indent=2)
        os.replace(temporary, json_path)

        marker = {
            "format": DSV4_W8A8_LOAD_FORMAT,
            "version": 1,
            "tensor_parallel_size": get_tensor_model_parallel_world_size(),
            "weight_layout": "logical_nd",
            "expert_quantization": "w8a8_dynamic_per_row",
            "linear_quantization": "w8a8_dynamic_per_row",
        }
        marker_path = output_dir / DSV4_W8A8_MARKER
        marker_tmp = marker_path.with_suffix(".json.tmp")
        with marker_tmp.open("w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2)
        os.replace(marker_tmp, marker_path)


def register_dsv4_w8a8_loader() -> None:
    """Register the preconverted loader exactly once in each process."""
    global _LOADER_REGISTERED
    if _LOADER_REGISTERED:
        return
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader(DSV4_W8A8_LOAD_FORMAT)(ShardedStateLoader310)
    _LOADER_REGISTERED = True
