"""GPU correctness tests for SM90 block-32x32 FP8 Triton GEMM kernels."""

import unittest

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.gemm.fp8_block32_sm90 import (
    _fp8_block32_splitk_reduce_kernel,
    _fp8_block32_swapab64_wgmma_kernel,
    fp8_block32_sm90_gemm,
    is_h20_sm90,
    use_fp8_block32_sm90_gemm,
)
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _ref_block32_fp8_gemm(
    a: torch.Tensor, b: torch.Tensor, a_s: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[0]
    a_f = (a.float().view(m, k // 32, 32) * a_s.unsqueeze(-1)).view(m, k)
    b_f = (b.float().view(n // 32, 32, k // 32, 32) * b_s[:, None, :, None]).view(n, k)
    return (a_f @ b_f.t()).to(torch.bfloat16)


def _make_block32_inputs(m: int, n: int, k: int, *, ue8m0: bool = True, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a_f = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=gen) * 0.5
    b_f = torch.randn((n, k), device="cuda", dtype=torch.float32, generator=gen) * 0.5
    a = a_f.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    b = b_f.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if ue8m0:
        exp_a = torch.randint(-6, 3, (m, k // 32), device="cuda", generator=gen)
        exp_b = torch.randint(-6, 3, (n // 32, k // 32), device="cuda", generator=gen)
        a_s = torch.pow(2.0, exp_a.float())
        b_s = torch.pow(2.0, exp_b.float())
    else:
        a_s = (
            torch.rand((m, k // 32), device="cuda", dtype=torch.float32, generator=gen)
            * 0.5
            + 0.05
        )
        b_s = (
            torch.rand(
                (n // 32, k // 32), device="cuda", dtype=torch.float32, generator=gen
            )
            * 0.5
            + 0.05
        )
    return a, b, a_s, b_s


@unittest.skipUnless(
    current_arch().is_sm(9, 0), "fp8_block32_sm90_gemm requires an NVIDIA SM90 GPU"
)
class TestFp8Block32Sm90Gemm(CustomTestCase):
    def _assert_matches_ref(
        self, m: int, n: int, k: int, *, ue8m0: bool = True, seed=0
    ):
        a, b, a_s, b_s = _make_block32_inputs(m, n, k, ue8m0=ue8m0, seed=seed)
        self.assertTrue(
            use_fp8_block32_sm90_gemm(a, b, a_s, b_s, [32, 32], torch.bfloat16)
        )
        out = fp8_block32_sm90_gemm(a, b, a_s, b_s)
        ref = _ref_block32_fp8_gemm(a, b, a_s, b_s)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (m, n))
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_empty_m_returns_empty_bf16(self):
        a, b, a_s, b_s = _make_block32_inputs(0, 128, 64)
        out = fp8_block32_sm90_gemm(a, b, a_s, b_s)
        self.assertEqual(out.shape, (0, 128))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_gemv_m1_direct_and_splitk(self):
        # Covers sk=1 (direct store), sk>1 (split-K + reduce), K=32 (1 group),
        # odd k-groups (K=288 -> 9, K=160 -> 5), and arbitrary fp32 scales.
        shapes = [
            (64, 32),
            (576, 5120),
            (1792, 5120),
            (4096, 1280),
            (5120, 288),
            (5120, 1024),
            (128, 160),
        ]
        for idx, (n, k) in enumerate(shapes):
            with self.subTest(n=n, k=k, ue8m0=True):
                self._assert_matches_ref(1, n, k, ue8m0=True, seed=idx)
            with self.subTest(n=n, k=k, ue8m0=False):
                self._assert_matches_ref(1, n, k, ue8m0=False, seed=idx + 100)

    def test_swapab64_small_m_regimes(self):
        # Covers FULL_M True (16, 32, 64) and False (2, 7, 17, 40),
        # EXACT_EVEN_K True and False, sk=1 and sk>1, and K=32.
        cases = [
            (2, 64, 32),
            (7, 576, 5120),
            (16, 1792, 5120),
            (17, 1856, 5120),
            (32, 4096, 1280),
            (40, 5120, 1024),
            (40, 5120, 288),
            (64, 3200, 5120),
            (5, 128, 160),
        ]
        for idx, (m, n, k) in enumerate(cases):
            with self.subTest(m=m, n=n, k=k):
                self._assert_matches_ref(m, n, k, ue8m0=(idx % 2 == 0), seed=idx + 200)

    def test_swapab64_strided_activation_view(self):
        m, n, k, full_k = 13, 512, 1024, 1792
        a_full, _, a_s_full, _ = _make_block32_inputs(m, n, full_k, seed=301)
        _, b, _, b_s = _make_block32_inputs(m, n, k, seed=302)
        a_view = a_full[:, :k]
        a_s_view = a_s_full[:, : k // 32]
        self.assertFalse(a_view.is_contiguous())
        self.assertTrue(
            use_fp8_block32_sm90_gemm(
                a_view, b, a_s_view, b_s, [32, 32], torch.bfloat16
            )
        )
        out = fp8_block32_sm90_gemm(a_view, b, a_s_view, b_s)
        ref = _ref_block32_fp8_gemm(a_view.contiguous(), b, a_s_view.contiguous(), b_s)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_swapab64_empty_trailing_split_does_not_double_count(self):
        # NUM_K_GROUPS=5 with SPLIT_K=4 yields k_groups_per_split=2 so pid_k=3 has
        # k_group_start=6 > k_group_end=5; it must accumulate 0 instead of rereading k_group=4.
        m, n, k, split_k = 8, 128, 160, 4
        a, b, a_s, b_s = _make_block32_inputs(m, n, k, seed=401)
        partials = torch.empty((split_k, m, n), dtype=torch.float32, device="cuda")
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        _fp8_block32_swapab64_wgmma_kernel[(n // 64, split_k)](
            a,
            b,
            a_s,
            b_s,
            partials,
            m,
            m * n,
            N=n,
            K=k,
            NUM_K_GROUPS=k // 32,
            stride_a_m=a.stride(0),
            stride_a_scale_m=a_s.stride(0),
            stride_b_scale_n=b_s.stride(0),
            SPLIT_K=split_k,
            BLOCK_M=16,
            DIRECT_STORE=False,
            EXACT_EVEN_K=False,
            FULL_M=False,
            num_warps=4,
            num_stages=4,
        )
        _fp8_block32_splitk_reduce_kernel[(m, n // 64)](
            partials, out, m * n, N=n, SPLIT_K=split_k, BLOCK_N=64, num_warps=2
        )
        ref = _ref_block32_fp8_gemm(a, b, a_s, b_s)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_large_m_both_block_n_and_splitk_paths(self):
        # Covers BLOCK_N=128 (N>=1024 or high-M narrow N=512) and BLOCK_N=64,
        # split-K when tile count < sm_count, direct store when >= sm_count,
        # and even/odd k-groups (including K=32, K=160, K=288).
        cases = [
            (65, 64, 32),
            (96, 576, 5120),
            (128, 512, 160),
            (128, 1792, 5120),
            (129, 1856, 5120),
            (192, 1024, 288),
            (257, 4096, 1280),
            (300, 5120, 1024),
            (2048, 512, 5120),
        ]
        for idx, (m, n, k) in enumerate(cases):
            with self.subTest(m=m, n=n, k=k):
                self._assert_matches_ref(m, n, k, ue8m0=(idx % 2 == 0), seed=idx + 500)

    def test_torch_compile_dynamic_m(self):
        @torch.compile(fullgraph=True, dynamic=True)
        def _compiled(a, b, a_s, b_s):
            return fp8_block32_sm90_gemm(a, b, a_s, b_s)

        n, k = 1024, 1024
        _, b, _, b_s = _make_block32_inputs(1, n, k, seed=650)
        for m in (1, 16, 128):
            with self.subTest(m=m):
                a, _, a_s, _ = _make_block32_inputs(m, n, k, seed=650 + m)
                out = _compiled(a, b, a_s, b_s)
                ref = _ref_block32_fp8_gemm(a, b, a_s, b_s)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_w8a8_block_fp8_matmul_triton_end_to_end(self):
        for m in (1, 16, 40, 64, 130):
            with self.subTest(m=m):
                a, b, a_s, b_s = _make_block32_inputs(m, 1024, 512, seed=600 + m)
                out = w8a8_block_fp8_matmul_triton(
                    a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
                )
                direct = fp8_block32_sm90_gemm(a, b, a_s, b_s)
                ref = _ref_block32_fp8_gemm(a, b, a_s, b_s)
                torch.testing.assert_close(out, direct, rtol=0, atol=0)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_capture_and_replay(self):
        for m in (1, 16, 128):
            with self.subTest(m=m):
                n, k = 1024, 1024
                a, b, a_s, b_s = _make_block32_inputs(m, n, k, seed=700 + m)
                # Warm up Triton compilation outside graph capture.
                _ = fp8_block32_sm90_gemm(a, b, a_s, b_s)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = fp8_block32_sm90_gemm(a, b, a_s, b_s)

                a2, b2, a_s2, b_s2 = _make_block32_inputs(m, n, k, seed=800 + m)
                a.copy_(a2)
                b.copy_(b2)
                a_s.copy_(a_s2)
                b_s.copy_(b_s2)
                graph.replay()
                torch.cuda.synchronize()

                ref = _ref_block32_fp8_gemm(a2, b2, a_s2, b_s2)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_large_strided_a_offset_above_int32_limit(self):
        # Strided A with small numel (32769 * 32) whose final row offset is
        # 32768 * 65536 == 2**31 must select 64-bit addressing instead of wrapping.
        m, k, n, row_stride = 32769, 32, 64, 65536
        backing = torch.zeros((m, row_stride), dtype=torch.float8_e4m3fn, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(901)
        vals = (
            torch.randn((m, k), dtype=torch.float32, device="cuda", generator=gen) * 0.5
        ).clamp(-448.0, 448.0)
        backing[:, :k].copy_(vals.to(torch.float8_e4m3fn))
        a_view = backing[:, :k]
        a_s = torch.ones((m, 1), dtype=torch.float32, device="cuda")
        _, b, _, b_s = _make_block32_inputs(1, n, k, seed=902)

        out = fp8_block32_sm90_gemm(a_view, b, a_s, b_s)
        # Spot-check boundary rows around and past the 2**31 offset boundary.
        for row in (0, 1, m - 2, m - 1):
            ref_row = _ref_block32_fp8_gemm(
                a_view[row : row + 1].contiguous(), b, a_s[row : row + 1], b_s
            )
            torch.testing.assert_close(
                out[row : row + 1], ref_row, rtol=2e-2, atol=2e-2
            )

    def test_deepgemm_hybrid_linear_matches_deepgemm_on_h20(self):
        from unittest.mock import patch

        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.quantization import fp8_utils

        if not (deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and is_h20_sm90()):
            self.skipTest("Requires H20 with JIT DeepGEMM enabled")

        cases = [
            (1, 576, 5120, True, False, False),
            (8, 1792, 5120, True, True, True),
            (16, 5120, 288, False, False, False),
            (64, 512, 5120, True, True, False),
            (128, 576, 5120, True, False, False),
        ]
        for idx, (m, n, k, ue8m0, use_bias, use_out) in enumerate(cases):
            with self.subTest(
                m=m, n=n, k=k, ue8m0=ue8m0, use_bias=use_bias, use_out=use_out
            ):
                gen = torch.Generator(device="cuda").manual_seed(950 + idx)
                x = (
                    torch.randn(
                        (m, k), dtype=torch.bfloat16, device="cuda", generator=gen
                    )
                    * 0.5
                )
                _, b, _, b_s = _make_block32_inputs(1, n, k, ue8m0=True, seed=960 + idx)
                bias = (
                    torch.randn(
                        (n,), dtype=torch.bfloat16, device="cuda", generator=gen
                    )
                    * 0.1
                    if use_bias
                    else None
                )
                out_buf = (
                    torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                    if use_out
                    else None
                )
                out_hybrid = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
                    x,
                    b,
                    [32, 32],
                    b_s,
                    bias=bias,
                    act_scale_ue8m0=ue8m0,
                    output=out_buf,
                )
                if use_out:
                    self.assertIs(out_hybrid, out_buf)
                with patch.object(
                    fp8_utils, "_use_fp8_block32_sm90_hybrid", return_value=False
                ):
                    out_dg = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
                        x, b, [32, 32], b_s, bias=bias, act_scale_ue8m0=ue8m0
                    )
                torch.testing.assert_close(out_hybrid, out_dg, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
