# SPDX-License-Identifier: Apache-2.0
"""Experimental MLA extension point for Ascend 310P.

The first milestone deliberately reuses the shared Ascend MLA implementation
while giving 310P a distinct backend class. Hardware-specific fallbacks can now
be added here without changing A2/A3 behavior.
"""

from vllm.logger import logger

from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAImpl


class AscendMLAImpl310(AscendMLAImpl):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "Using the experimental Ascend 310P MLA implementation. "
            "Correctness and performance are not yet guaranteed."
        )
        super().__init__(*args, **kwargs)


class AscendMLABackend310(AscendMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA_310P"

    @staticmethod
    def get_impl_cls() -> type[AscendMLAImpl310]:
        return AscendMLAImpl310
