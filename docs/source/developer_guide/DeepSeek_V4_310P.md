# Experimental DeepSeek V4 backend for Ascend 310P

This fork contains an opt-in development path for running DeepSeek V4 on
Ascend 310P. It is not production ready.

Enable the backend with:

```bash
export VLLM_ASCEND_ENABLE_DSV4_310P=1
```

The initial implementation provides:

- dedicated 310P MLA and DSA backend classes;
- correct 310P attention backend routing for compressed MLA/DSA;
- reuse of the shared DeepSeek V4 KV-cache allocator and binding logic;
- compatibility with the 310P AllGather expert-parallel path in current main.

Remaining execution blockers are tracked in this order:

1. replace MXFP4 custom-dtype conversion with a 310P-compatible weight path;
2. provide composed torch-npu fallbacks for compressor metadata and compressor;
3. provide a 310P indexer/top-k implementation;
4. provide sparse prefill/decode attention fallback;
5. replace the fallbacks with tuned AscendC kernels.

The feature gate intentionally leaves the default upstream 310P behavior
unchanged. Without it, MLA and DeepSeek Sparse Attention still fail fast.
