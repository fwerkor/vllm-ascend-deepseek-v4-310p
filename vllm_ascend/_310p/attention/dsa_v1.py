# SPDX-License-Identifier: Apache-2.0
"""Experimental DeepSeek Sparse Attention extension point for Ascend 310P.

This class initially shares the model-level implementation with the main
Ascend backend. Unsupported fused operators will be replaced incrementally by
310P ACLNN/AscendC or composed torch-npu fallbacks in this module.
"""

from vllm.logger import logger

from vllm_ascend.attention.dsa_v1 import AscendDSABackend, AscendDSAImpl


class AscendDSAImpl310(AscendDSAImpl):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "Using the experimental Ascend 310P DeepSeek Sparse Attention implementation. "
            "Unsupported fused operators are being replaced incrementally."
        )
        super().__init__(*args, **kwargs)


class AscendDSABackend310(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_310P"

    @staticmethod
    def get_impl_cls() -> type[AscendDSAImpl310]:
        return AscendDSAImpl310
