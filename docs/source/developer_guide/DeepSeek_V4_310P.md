# Experimental DeepSeek V4 backend for Ascend 310P

This fork contains an opt-in development path for running DeepSeek V4 on
Ascend 310P. It is not production ready.

Enable the backend with:

```bash
export VLLM_ASCEND_ENABLE_DSV4_310P=1
```

The initial implementation provides:

- dedicated 310P MLA and DSA backend classes;
- model-specific routing of every DeepSeek V4 layer to the DSA backend, which
  handles the per-layer compression ratios 1, 4, and 128;
- reuse of the shared DeepSeek V4 KV-cache allocator and binding logic;
- compatibility with the 310P AllGather expert-parallel path in current main.
- a streaming expert fallback that keeps local EP shards in packed MXFP4 and
  materializes only the currently executing decoder layer as native 310P
  per-row W8A8;
- an optional eager diagnostic mode, selected with
  `VLLM_ASCEND_DSV4_310P_EXPERT_MODE=eager_w8a8`.
- composed Hyper-Connection pre/Sinkhorn/post fallbacks based on the official
  DeepSeek V4 reference formulas.
- logical-shape-preserving W8A8 allocation so FRACTAL_NZ conversion retains
  the expert dimension required by the 310P quantized grouped-matmul kernel.
- software E4M3FN decoding and block-FP8 to per-row INT8 conversion for all
  DeepSeek V4 dense and shared-expert linear layers.
- a 310P-compatible shared-expert path that emits FP16 gate/up activations,
  then composes clamped SwiGLU and dynamic INT8 quantization. This avoids the
  unsupported INT32 output mode of WeightNZ QuantMatmul.

Remaining execution blockers are tracked in this order:

1. provide composed torch-npu fallbacks for compressor metadata and compressor;
2. provide a 310P indexer/top-k implementation;
3. provide sparse prefill/decode attention fallback;
4. replace full-layer streaming conversion with active-expert conversion;
5. replace the fallbacks with tuned AscendC kernels.

The feature gate intentionally leaves the default upstream 310P behavior
unchanged. Without it, MLA and DeepSeek Sparse Attention still fail fast.
