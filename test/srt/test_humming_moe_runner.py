import importlib
import sys
import unittest

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.test_utils import CustomTestCase


class _DummyLayer(torch.nn.Module):
    params_dtype = torch.bfloat16


class TestHummingMoeRunner(CustomTestCase):
    def test_humming_moe_import_does_not_load_multimodal_gen(self):
        module_name = "sglang.srt.layers.moe.fused_moe_triton.moe_fused_mul_sum"
        sys.modules.pop(module_name, None)
        before = set(sys.modules)

        importlib.import_module(module_name)

        new_multimodal_modules = [
            module
            for module in sys.modules
            if module.startswith("sglang.multimodal_gen") and module not in before
        ]
        self.assertEqual([], new_multimodal_modules)

    def test_none_a2a_reuses_runner_core(self):
        old_backend = moe_utils.MOE_A2A_BACKEND
        try:
            moe_utils.MOE_A2A_BACKEND = MoeA2ABackend.NONE
            runner = MoeRunner(
                MoeRunnerBackend.HUMMING,
                MoeRunnerConfig(
                    num_experts=1,
                    num_local_experts=1,
                    layer=_DummyLayer(),
                ),
            )
            self.assertIsNone(runner.fused_func)
            self.assertIsNotNone(runner.runner_core)
        finally:
            moe_utils.MOE_A2A_BACKEND = old_backend


if __name__ == "__main__":
    unittest.main()
