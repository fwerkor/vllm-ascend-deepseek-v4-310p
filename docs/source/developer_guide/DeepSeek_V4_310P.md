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
- loading-time conversion of each local EP expert shard from packed MXFP4 to
  the native 310P per-row W8A8 MoE path.

Remaining execution blockers are tracked in this order:

1. provide composed torch-npu fallbacks for compressor metadata and compressor;
2. provide a 310P indexer/top-k implementation;
3. provide sparse prefill/decode attention fallback;
4. validate W8A8 memory headroom with EP=8 and reduce conversion workspace;
5. replace the fallbacks with tuned AscendC kernels.

The feature gate intentionally leaves the default upstream 310P behavior
unchanged. Without it, MLA and DeepSeek Sparse Attention still fail fast.
