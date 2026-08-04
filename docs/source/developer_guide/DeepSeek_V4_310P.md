# Experimental DeepSeek V4 backend for Ascend 310P

This fork provides an opt-in DeepSeek V4-Flash inference path for Ascend 310P.
It is intended for bring-up, validation, and further kernel optimization. The
standard upstream 310P behavior remains unchanged unless the feature gate is
enabled.

## Validated configuration

The current implementation has been validated with:

- eight Ascend 310P3 devices;
- `quay.io/ascend/vllm-ascend:nightly-main-310p`;
- CANN 9.1.0 beta;
- tensor parallel size 8 and expert parallel size 8;
- FP16 activations;
- a DeepSeek V4-Flash checkpoint containing block-FP8 dense weights and
  MXFP4 routed-expert weights;
- a maximum model length of 128 tokens.

Enable the model-specific backend and eager 310P expert conversion:

```bash
export VLLM_ASCEND_ENABLE_DSV4_310P=1
export VLLM_ASCEND_DSV4_310P_EXPERT_MODE=eager_w8a8
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

A validated server command is:

```bash
vllm serve /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype float16 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 8 \
  --kv-cache-memory 134217728 \
  --block-size 32 \
  --enforce-eager \
  --load-format auto \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice
```

## Implemented paths

The 310P backend currently provides:

- eager MXFP4-to-W8A8 conversion for local routed-expert shards;
- block-FP8-to-W8A8 loading for unsupported 310P linear paths;
- AllGather expert parallelism with global-to-local expert remapping;
- composed 310P Hyper-Connection and clipped SwiGLU implementations;
- dense short-context DeepSeek sparse-attention fallback;
- interleaved DeepSeek V4 RoPE and inverse-RoPE;
- physical/logical hybrid block-table decoding for paged SWA KV cache;
- 310P-safe cache allocation, worker initialization, and memory cleanup.

The hybrid block-table conversion is required because a physical 32-token KV
page can be represented as multiple smaller logical kernel blocks. The fallback
converts logical block IDs and offsets back to physical cache coordinates and
performs strict bounds checking before gathering cached keys.

## Validation

The repository regression suite used for this path covers 310P model routing,
KV-cache allocation and block tables, expert selection, AllGather dispatch and
combine, W8A8 grouped matmul, clipped SwiGLU, and shared/routed expert output.
The validated suite result is:

```text
141 passed, 2 skipped
```

Greedy generation was also checked with Chinese, English, arithmetic, code,
sequential requests, and two concurrent requests. No worker error, OOM, or
cross-request cache corruption was observed.

## Current limitations

- The attention implementation is a correctness-oriented dense fallback and is
  exact only while `max_model_len` does not exceed the model sliding window.
- The validated setup is limited to 128 tokens by the current memory budget and
  has not been qualified for long-context serving.
- Routed experts are converted eagerly during startup. On the validated
  eight-device server, full model loading and conversion takes about eight
  minutes.
- Greedy decode throughput is currently about 1.0 to 1.2 output tokens per
  second for a single short request. Tuned AscendC attention and quantization
  kernels are still required for production performance.
- Prefix caching, speculative decoding, KV transfer, and disaggregated serving
  have not been validated on this 310P path.
