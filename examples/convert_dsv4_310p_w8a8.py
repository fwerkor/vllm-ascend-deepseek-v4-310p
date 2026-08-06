# SPDX-License-Identifier: Apache-2.0
"""Convert a DeepSeek V4 checkpoint to TP-sharded Ascend 310P W8A8."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from vllm import LLM

from vllm_ascend._310p.sharded_state_loader_310p import (
    DSV4_W8A8_LOAD_FORMAT,
    DSV4_W8A8_MARKER,
)

_WEIGHT_SUFFIXES = {".bin", ".pt", ".safetensors"}
_SKIP_METADATA = {"model.safetensors.index.json", DSV4_W8A8_MARKER, "parameters_type_map.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-file-size", type=int, default=4 * 1024**3)
    parser.add_argument("--max-model-len", type=int, default=128)
    return parser.parse_args()


def copy_metadata(source: Path, output: Path) -> None:
    for item in source.iterdir():
        if item.name in _SKIP_METADATA or item.name == ".git":
            continue
        if item.is_file() and item.suffix in _WEIGHT_SUFFIXES:
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def validate_checkpoint(output: Path, tensor_parallel_size: int) -> tuple[int, int]:
    marker_path = output / DSV4_W8A8_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("format") != DSV4_W8A8_LOAD_FORMAT:
        raise RuntimeError(f"Unexpected checkpoint marker: {marker!r}")
    if int(marker.get("tensor_parallel_size", -1)) != tensor_parallel_size:
        raise RuntimeError(f"Unexpected checkpoint TP size: {marker!r}")

    total_files = 0
    total_bytes = 0
    for rank in range(tensor_parallel_size):
        rank_files = sorted(output.glob(f"model-rank-{rank}-part-*.safetensors"))
        if not rank_files:
            raise RuntimeError(f"Missing sharded checkpoint files for TP rank {rank}.")
        for path in rank_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                if not list(f.keys()):
                    raise RuntimeError(f"Empty safetensors shard: {path}")
            total_files += 1
            total_bytes += path.stat().st_size
    return total_files, total_bytes


def main() -> None:
    args = parse_args()
    source = args.model.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Source and output directories must differ.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=str(source),
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        kv_cache_memory_bytes=128 * 1024**2,
        block_size=32,
        enforce_eager=True,
        load_format="auto",
        tokenizer_mode="deepseek_v4",
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
    )
    llm.llm_engine.engine_core.save_sharded_state(
        path=str(output),
        max_size=args.max_file_size,
    )
    copy_metadata(source, output)
    files, size = validate_checkpoint(output, args.tensor_parallel_size)
    print(f"validated {files} rank shards totaling {size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
