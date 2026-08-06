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
export VLLM_ASCEND_DSV4_310P_DETERMINISTIC=1
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

`VLLM_ASCEND_DSV4_310P_DETERMINISTIC=1` enables deterministic HCCL and
ACLNN execution before distributed communicators and model operators are
created. This is recommended for greedy evaluation, reproducible DSpark
acceptance, and regression testing. It remains opt-in at the backend level so
other 310P workloads retain their existing execution policy.

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
- opt-in deterministic HCCL and ACLNN initialization for reproducible decoding;
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
261 passed, 3 skipped
```

The skipped cases require an NPU-enabled PyTorch build and are not executable in
the CPU-only unit-test container.

A real TP8/EP8 deterministic service was validated with greedy English,
arithmetic, sequence-completion, code-related, and repeated requests:

- six identical 32-token DSpark requests produced identical token IDs and the
  same speculative path each time: 50 drafted and 25 accepted tokens;
- two consecutive fixed three-prompt benchmarks produced identical outputs and
  identical aggregate acceptance, 137 accepted tokens out of 285 drafted
  tokens, or 48.07%;
- those two 192-token benchmark rounds completed at 3.622 and 3.607 output
  tokens/s respectively;
- eight consecutive 16-token requests completed successfully, with 128 output
  tokens in 47.692 seconds, or 2.684 output tokens/s in aggregate;
- target-only checks also reproduced identical top-5 log probabilities across
  six 8-token requests and identical outputs across three 32-token requests;
- health remained HTTP 200 and the logs contained no AICore exception,
  out-of-range access, NaN, worker failure, or engine shutdown.

Without deterministic execution, near-tied logits could choose different greedy
tokens across identical requests and consequently change DSpark acceptance.
The measured top-two log-probability margins at the first divergence were only
0.016 to 0.125. Enabling deterministic execution removed both token and
acceptance-path variation without a measurable throughput penalty in this
configuration.

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
- Full TP8/EP8 target and draft loading takes about eight minutes. Each rank holds
  approximately 37.8 GiB of model weights, leaving about 2.3 GiB free after
  startup cleanup on the validated server.
- DSpark acceptance remains workload-dependent, and later positions in the
  five-token block are accepted less frequently. Deterministic mode makes the
  acceptance path reproducible for an identical request.
- The current path uses eager execution. Tuned AscendC attention, grouped-matmul,
  and graph-mode kernels are still required for production performance.
- Prefix caching, KV transfer, disaggregated serving, multi-request concurrency,
  and contexts longer than 128 tokens have not been validated on this path.
