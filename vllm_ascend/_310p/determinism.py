# SPDX-License-Identifier: Apache-2.0

import os

import torch_npu
from vllm.logger import logger

_DSV4_DETERMINISTIC_ENV = "VLLM_ASCEND_DSV4_310P_DETERMINISTIC"


def configure_dsv4_determinism() -> None:
    if os.getenv(_DSV4_DETERMINISTIC_ENV, "0") != "1":
        return

    # HCCL reads this before communicator initialization. ACLNN reads the
    # PyTorch deterministic state when operators are dispatched.
    os.environ["HCCL_DETERMINISTIC"] = "true"
    torch_npu.npu.set_deterministic_level(1)
    logger.info_once(
        "Enabled deterministic HCCL and ACLNN execution for DeepSeek V4 on Ascend 310P.",
        scope="local",
    )
