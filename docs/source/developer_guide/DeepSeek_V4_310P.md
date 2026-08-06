# Experimental DeepSeek V4 backend for Ascend 310P

This fork provides an opt-in DeepSeek V4-Flash inference path for Ascend 310P.
It is intended for bring-up, correctness validation, and further kernel
optimization. Standard upstream 310P behavior remains unchanged unless the
feature gate is enabled.

## Validated configuration

The current implementation has been validated with:

- eight Ascend 310P3 devices;
- `quay.io/ascend/vllm-ascend:nightly-main-310p`;
- CANN 9.1.0 beta;
- tensor parallel size 8 and expert parallel size 8;
- FP16 activations;
- a preconverted W8A8 target checkpoint;
- the original DeepSeek V4-Flash checkpoint as the DSpark draft source;
- a maximum model length of 128 tokens;
- one active sequence and a 128 MiB KV-cache budget.

The original checkpoint contains block-FP8 dense tensors and packed MXFP4
routed experts. The target model is converted ahead of serving, while the three
DSpark draft layers are loaded from the original checkpoint and converted per
local expert shard during startup.

Enable the model-specific backend and select the preconverted target layout:

```bash
export VLLM_ASCEND_ENABLE_DSV4_310P=1
export VLLM_ASCEND_DSV4_310P_EXPERT_MODE=preconverted_w8a8
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

Convert a checkpoint with the provided utility before serving:

```bash
python examples/convert_dsv4_310p_w8a8.py \
  --model /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-W8A8-310P \
  --tensor-parallel-size 8
```

A validated server command is:

```bash
vllm serve /path/to/DeepSeek-V4-Flash-W8A8-310P \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype float16 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --kv-cache-memory 134217728 \
  --block-size 32 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --load-format dsv4_310p_w8a8 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --speculative-config \
  '{
    "method": "dspark",
    "model": "/path/to/DeepSeek-V4-Flash",
    "num_speculative_tokens": 5,
    "enforce_eager": true,
    "draft_load_config": {"load_format": "auto"}
  }'
```

DSpark uses a five-token native draft block. Values below five are rejected
because truncating that block produces incorrect output.

## Implemented paths

The 310P backend currently provides:

- preconverted target W8A8 loading with per-rank TP/EP shards;
- packed MXFP4-to-W8A8 conversion for local DSpark routed experts;
- bit-preserving E8M0 expert-scale loading;
- block-FP8-to-W8A8 loading for unsupported 310P linear paths;
- AllGather expert parallelism with global-to-local expert remapping;
- composed 310P Hyper-Connection and clipped SwiGLU implementations;
- a dense short-context DeepSeek sparse-attention fallback;
- interleaved DeepSeek V4 RoPE and inverse-RoPE;
- paged SWA cache writes without unavailable 310P custom operators;
- physical/logical hybrid block-table decoding for paged KV cache;
- DSpark draft embedding, LM-head, and index-buffer sharing;
- 310P-safe cache allocation, worker initialization, CPU affinity, and memory
  cleanup.

Raw MXFP4 and converted W8A8 weights both use byte-sized storage. The loader
therefore identifies the layout from tensor widths and scale shape/dtype rather
than trusting a process-wide mode alone. E8M0 scale tensors are copied in their
checkpoint float8 dtype when the destination is also E8M0; converting the source
to `uint8` before `copy_` would perform a numerical cast and corrupt the exponent
bytes.

The hybrid block-table conversion is required because a physical 32-token KV
page can be represented as multiple smaller logical kernel blocks. The fallback
converts logical block IDs and offsets back to physical cache coordinates and
performs bounds checking before gathering cached keys.

## Validation

The broad regression command used for this path covers all 310P unit tests plus
DSpark proposer, CPU binding, and dynamic W8A8 paths. The current result is:

```text
259 passed, 3 skipped
```

The skipped cases require an NPU-enabled PyTorch build and are not executable in
the CPU-only unit-test container.

A real TP8/EP8 service was also validated with greedy English, arithmetic,
sequence-completion, code-related, and repeated sequential requests:

- eight consecutive 16-token requests completed successfully;
- 128 output tokens completed in 47.266 seconds in that sequential run, or
  2.708 output tokens/s in aggregate;
- a separate 64-token request completed in 27.514 seconds, or 2.326 output
  tokens/s;
- the cumulative DSpark draft acceptance observed after the validation requests
  was 129 accepted tokens out of 395 drafted tokens, or 32.7%;
- health remained HTTP 200 and the logs contained no AICore exception,
  out-of-range access, NaN, worker failure, or engine shutdown.

These figures are bring-up measurements for one short-context request at a time,
not production service-level guarantees. Prompt lengths and cache state affect
the reported throughput.

## Current limitations

- The attention implementation is a correctness-oriented dense fallback and is
  exact only while `max_model_len` does not exceed the model sliding window.
- The validated setup is limited to 128 tokens by the current memory budget and
  has not been qualified for long-context serving.
- The validated 128 MiB KV-cache allocation provides 203 tokens of capacity and
  approximately 1.59x theoretical concurrency at a 128-token request length;
  production concurrency has not been qualified.
- Full TP8/EP8 target and draft loading takes about 7.5 minutes. Each rank holds
  approximately 37.8 GiB of model weights, leaving about 2.3 GiB free after
  startup cleanup on the validated server.
- DSpark is functional and improves accepted-token throughput, but its acceptance
  rate is workload-dependent and later positions in the five-token block are
  accepted less frequently.
- The current path uses eager execution. Tuned AscendC attention, grouped-matmul,
  and graph-mode kernels are still required for production performance.
- Prefix caching, KV transfer, disaggregated serving, multi-request concurrency,
  and contexts longer than 128 tokens have not been validated on this path.
