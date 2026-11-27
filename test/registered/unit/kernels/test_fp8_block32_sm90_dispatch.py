"""CPU unit tests for SM90 block-32x32 FP8 GEMM dispatch, gating, and SM scaling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels as K
import sglang.kernels.arch as arch_module
from sglang.kernels import KernelBackend, PlatformInfo
from sglang.kernels.arch._cuda import CudaArch
from sglang.kernels.arch._hip import HipArch
from sglang.kernels.arch.base import DeviceArch
from sglang.kernels.ops.gemm import fp8_block32_sm90, fp8_kernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cuda_arch(
    major: int, minor: int, num_sms: int = 78, name: str = "NVIDIA H20"
) -> CudaArch:
    arch = CudaArch()
    arch.props = SimpleNamespace(
        major=major, minor=minor, multi_processor_count=num_sms, name=name
    )
    return arch


def _hip_arch(gcn_arch: str = "gfx950") -> HipArch:
    arch = HipArch()
    arch.props = SimpleNamespace(gcnArchName=gcn_arch, multi_processor_count=256)
    return arch


def _fake_tensor(
    shape,
    *,
    dtype,
    strides=None,
    is_cuda=True,
    device="cuda:0",
    contiguous=True,
):
    if strides is None:
        s = 1
        rev = []
        for dim in reversed(shape):
            rev.append(s)
            s *= dim
        strides = tuple(reversed(rev))

    def _stride(dim=None):
        return strides if dim is None else strides[dim]

    return SimpleNamespace(
        shape=tuple(shape),
        ndim=len(shape),
        dtype=dtype,
        is_cuda=is_cuda,
        device=device,
        stride=_stride,
        is_contiguous=lambda: contiguous,
    )


def _valid_fake_operands(m=16, n=5120, k=1024, **overrides):
    a_strides = overrides.pop("a_strides", None)
    as_strides = overrides.pop("as_strides", None)
    bs_strides = overrides.pop("bs_strides", None)
    b_contiguous = overrides.pop("b_contiguous", True)
    device = overrides.pop("device", "cuda:0")
    b_device = overrides.pop("b_device", device)
    is_cuda = overrides.pop("is_cuda", True)
    a_dtype = overrides.pop("a_dtype", torch.float8_e4m3fn)
    b_dtype = overrides.pop("b_dtype", torch.float8_e4m3fn)
    as_dtype = overrides.pop("as_dtype", torch.float32)
    bs_dtype = overrides.pop("bs_dtype", torch.float32)
    a_shape = overrides.pop("a_shape", (m, k))
    b_shape = overrides.pop("b_shape", (n, k))
    as_shape = overrides.pop("as_shape", (m, k // 32))
    bs_shape = overrides.pop("bs_shape", (n // 32, k // 32))
    assert not overrides, f"Unexpected overrides: {overrides}"

    return (
        _fake_tensor(
            a_shape,
            dtype=a_dtype,
            strides=a_strides,
            is_cuda=is_cuda,
            device=device,
            contiguous=(a_strides is None),
        ),
        _fake_tensor(
            b_shape,
            dtype=b_dtype,
            is_cuda=is_cuda,
            device=b_device,
            contiguous=b_contiguous,
        ),
        _fake_tensor(
            as_shape,
            dtype=as_dtype,
            strides=as_strides,
            is_cuda=is_cuda,
            device=device,
        ),
        _fake_tensor(
            bs_shape,
            dtype=bs_dtype,
            strides=bs_strides,
            is_cuda=is_cuda,
            device=device,
        ),
    )


class TestFp8Block32Sm90RegistryAndArch(CustomTestCase):
    def test_registry_spec_is_sm90_triton_only(self):
        spec = K.select_kernel(
            "gemm.fp8_block32_sm90_gemm", backend=KernelBackend.TRITON
        )
        self.assertEqual(
            spec.target,
            "sglang.kernels.ops.gemm.fp8_block32_sm90:fp8_block32_sm90_gemm",
        )
        self.assertTrue(
            spec.is_available(
                PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
            )
        )
        for platform in (
            PlatformInfo(device_type="cuda", cuda_arch_major=8, cuda_arch_minor=9),
            PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=1),
            PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0),
            PlatformInfo(device_type="cuda", cuda_arch_major=12, cuda_arch_minor=0),
            PlatformInfo(device_type="hip"),
            PlatformInfo(device_type="musa"),
            PlatformInfo(device_type="cpu"),
        ):
            self.assertFalse(spec.is_available(platform), platform)

    def test_arch_gate_accepts_only_nvidia_sm90(self):
        operands = _valid_fake_operands()
        cases = [
            (_cuda_arch(9, 0, 78), True),
            (_cuda_arch(9, 0, 132), True),
            (_cuda_arch(8, 9, 128), False),
            (_cuda_arch(9, 1, 132), False),
            (_cuda_arch(10, 0, 148), False),
            (_cuda_arch(12, 0, 170), False),
            (_hip_arch("gfx950"), False),
            (DeviceArch(), False),
        ]
        for arch, expected in cases:
            with (
                self.subTest(arch=arch),
                patch.object(arch_module, "_DEVICE_ARCH", arch),
            ):
                self.assertIs(
                    fp8_block32_sm90.use_fp8_block32_sm90_gemm(
                        *operands, [32, 32], torch.bfloat16
                    ),
                    expected,
                )


class TestFp8Block32Sm90SplitKHeuristic(CustomTestCase):
    def test_split_k_scales_with_sm_count(self):
        # SwapAB64 (BLOCK_N=64): N=4096 (64 tiles), K=1280 (40 k-groups)
        self.assertEqual(fp8_block32_sm90._choose_split_k(64, 40, sm_count=78), 2)
        self.assertEqual(fp8_block32_sm90._choose_split_k(64, 40, sm_count=132), 4)

        # SwapAB64: N=8192 (128 tiles), K=5120 (160 k-groups)
        # 128 tiles already exceeds 1.2 * 78 on H20, but benefits from sk=2 on 132-SM H200.
        self.assertEqual(fp8_block32_sm90._choose_split_k(128, 160, sm_count=78), 1)
        self.assertEqual(fp8_block32_sm90._choose_split_k(128, 160, sm_count=132), 2)

        # GEMV (BLOCK_N=32): N=1792 (56 tiles), K=5120 (160 k-groups)
        self.assertEqual(fp8_block32_sm90._choose_split_k(56, 160, sm_count=78), 3)
        self.assertEqual(fp8_block32_sm90._choose_split_k(56, 160, sm_count=114), 4)
        self.assertEqual(fp8_block32_sm90._choose_split_k(56, 160, sm_count=132), 5)

    def test_short_k_and_wide_n_stay_single_split(self):
        # down_proj (N=5120 -> 80 SwapAB tiles, K=288 -> 9 k-groups): too short to split.
        for sm in (78, 114, 132):
            self.assertEqual(fp8_block32_sm90._choose_split_k(80, 9, sm_count=sm), 1)
            self.assertEqual(fp8_block32_sm90._choose_split_k(160, 9, sm_count=sm), 1)
            # Wide N well past one wave avoids extra reduction launch.
            self.assertEqual(fp8_block32_sm90._choose_split_k(256, 160, sm_count=sm), 1)

    def test_boundary_invariants(self):
        for sm in (0, 1, 78, 114, 132):
            for num_n in (1, 9, 28, 64, 80, 160):
                for num_k in (1, 2, 5, 9, 32, 160):
                    sk = fp8_block32_sm90._choose_split_k(num_n, num_k, sm_count=sm)
                    self.assertGreaterEqual(sk, 1)
                    self.assertLessEqual(sk, num_k)
                    if sm <= 0 or num_k <= 1:
                        self.assertEqual(sk, 1)


class TestFp8Block32Sm90EligibilityAndValidation(CustomTestCase):
    def test_predicate_accepts_valid_variants_and_rejects_invalid(self):
        sm90 = _cuda_arch(9, 0, 78)
        with patch.object(arch_module, "_DEVICE_ARCH", sm90):
            # Tuple block_size, M=0, strided A with multiple-of-32 row stride, K=32 singleton scale dim.
            self.assertTrue(
                fp8_block32_sm90.use_fp8_block32_sm90_gemm(
                    *_valid_fake_operands(m=0, n=64, k=32), (32, 32), torch.bfloat16
                )
            )
            self.assertTrue(
                fp8_block32_sm90.use_fp8_block32_sm90_gemm(
                    *_valid_fake_operands(m=16, n=128, k=1024, a_strides=(1792, 1)),
                    [32, 32],
                    torch.bfloat16,
                )
            )
            self.assertTrue(
                fp8_block32_sm90.use_fp8_block32_sm90_gemm(
                    *_valid_fake_operands(
                        m=8, n=64, k=32, as_strides=(1, 8), bs_strides=(1, 2)
                    ),
                    [32, 32],
                    torch.bfloat16,
                )
            )

            rejected = [
                ("block_128", _valid_fake_operands(), [128, 128], torch.bfloat16),
                ("block_16_32", _valid_fake_operands(), [16, 32], torch.bfloat16),
                ("fp16_out", _valid_fake_operands(), [32, 32], torch.float16),
                ("fp32_out", _valid_fake_operands(), [32, 32], torch.float32),
                ("cpu", _valid_fake_operands(is_cuda=False), [32, 32], torch.bfloat16),
                (
                    "device_mismatch",
                    _valid_fake_operands(b_device="cuda:1"),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "bad_n_mod64",
                    _valid_fake_operands(n=96, bs_shape=(3, 32)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "bad_k_mod32",
                    _valid_fake_operands(k=48, as_shape=(16, 2), bs_shape=(160, 2)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "bad_a_stride0",
                    _valid_fake_operands(a_strides=(1040, 1)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "bad_a_stride1",
                    _valid_fake_operands(a_strides=(32, 2)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "non_contig_b",
                    _valid_fake_operands(b_contiguous=False),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "non_contig_as",
                    _valid_fake_operands(as_strides=(1, 16)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "int32_scales",
                    _valid_fake_operands(as_dtype=torch.int32),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "3d_a",
                    _valid_fake_operands(a_shape=(2, 8, 1024), as_shape=(2, 8, 32)),
                    [32, 32],
                    torch.bfloat16,
                ),
                (
                    "negative_a_stride",
                    _valid_fake_operands(a_strides=(-1024, 1)),
                    [32, 32],
                    torch.bfloat16,
                ),
            ]
            for name, operands, block_size, out_dtype in rejected:
                with self.subTest(case=name):
                    self.assertFalse(
                        fp8_block32_sm90.use_fp8_block32_sm90_gemm(
                            *operands, block_size, out_dtype
                        )
                    )

    def test_direct_entrypoint_raises_on_contract_violations(self):
        a = torch.empty((4, 64), dtype=torch.float8_e4m3fn)
        b = torch.empty((64, 64), dtype=torch.float8_e4m3fn)
        a_s = torch.empty((4, 2), dtype=torch.float32)
        b_s = torch.empty((2, 2), dtype=torch.float32)

        # CPU tensors must fail explicitly instead of silently falling back.
        with self.assertRaises(ValueError):
            fp8_block32_sm90.fp8_block32_sm90_gemm(a, b, a_s, b_s)

        # Shape/scale mismatches raise ValueError.
        with self.assertRaises(ValueError):
            fp8_block32_sm90.fp8_block32_sm90_gemm(
                a, torch.empty((96, 64), dtype=torch.float8_e4m3fn), a_s, b_s
            )
        with self.assertRaises(ValueError):
            fp8_block32_sm90.fp8_block32_sm90_gemm(
                a, b, torch.empty((4, 1), dtype=torch.float32), b_s
            )

    def test_public_entrypoint_rejects_non_sm90_cuda_devices(self):
        a, b, a_s, b_s = _valid_fake_operands(m=8, n=64, k=64, device="cuda:1")
        out = _fake_tensor((8, 64), dtype=torch.bfloat16, device="cuda:1")
        via_registry = K.get_kernel("gemm.fp8_block32_sm90_gemm")

        for cap in ((8, 0), (8, 9), (10, 0)):
            with (
                self.subTest(capability=cap),
                patch.object(torch.version, "hip", None),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=cap
                ) as mock_cap,
                patch.object(
                    fp8_block32_sm90,
                    "_fp8_block32_sm90_gemm_custom_op",
                    side_effect=AssertionError("launched on non-SM90 device"),
                ),
            ):
                with self.assertRaises(ValueError):
                    via_registry(a, b, a_s, b_s, out=out)
                mock_cap.assert_called_once_with("cuda:1")

        # ROCm reports CUDA-like devices and can return (9, 0) on gfx90a; must reject.
        with (
            patch.object(torch.version, "hip", "6.3"),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                fp8_block32_sm90,
                "_fp8_block32_sm90_gemm_custom_op",
                side_effect=AssertionError("launched on ROCm"),
            ),
        ):
            with self.assertRaises(ValueError):
                fp8_block32_sm90.fp8_block32_sm90_gemm(a, b, a_s, b_s, out=out)

        # NVIDIA SM90 passes validation and invokes the custom op.
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                fp8_block32_sm90, "_fp8_block32_sm90_gemm_custom_op"
            ) as mock_op,
        ):
            res = fp8_block32_sm90.fp8_block32_sm90_gemm(a, b, a_s, b_s, out=out)
            self.assertIs(res, out)
            mock_op.assert_called_once_with(out, a, b, a_s, b_s)

    def test_needs_int64_offsets_checks_actual_strided_extents(self):
        # Strided A with small numel but max element offset >= 2**31 must enable int64.
        a_strided, _, a_s_small, b_s_small = _valid_fake_operands(
            m=32769, n=64, k=32, a_strides=(65536, 1)
        )
        self.assertTrue(
            fp8_block32_sm90._needs_int64_offsets(
                a_strided, a_s_small, b_s_small, 32769, 64, 32, 1, 1, 64
            )
        )

        # Same logical shape contiguous stays on 32-bit offsets.
        a_contig, _, a_s_contig, b_s_contig = _valid_fake_operands(m=32769, n=64, k=32)
        self.assertFalse(
            fp8_block32_sm90._needs_int64_offsets(
                a_contig, a_s_contig, b_s_contig, 32769, 64, 32, 1, 1, 64
            )
        )

        # Strided As and Bs whose max offsets cross 2**31 must enable int64.
        _, _, a_s_strided, _ = _valid_fake_operands(
            m=32769, n=64, k=32, as_strides=(65536, 1)
        )
        self.assertTrue(
            fp8_block32_sm90._needs_int64_offsets(
                a_contig, a_s_strided, b_s_contig, 32769, 64, 32, 1, 1, 64
            )
        )

        _, _, _, b_s_strided = _valid_fake_operands(
            m=32, n=64, k=32, bs_strides=(2**31, 1)
        )
        self.assertTrue(
            fp8_block32_sm90._needs_int64_offsets(
                a_contig, a_s_contig, b_s_strided, 32, 64, 32, 1, 1, 64
            )
        )

        # Split-K partials buffer crossing 2**31 elements must enable int64.
        a_ok, _, a_s_ok, b_s_ok = _valid_fake_operands(m=16385, n=65536, k=64)
        self.assertTrue(
            fp8_block32_sm90._needs_int64_offsets(
                a_ok, a_s_ok, b_s_ok, 16385, 65536, 64, 2, 2, 64
            )
        )


class TestW8A8BlockFp8MatmulTritonDispatch(CustomTestCase):
    def test_sm90_fast_path_respects_determinism(self):
        a = torch.empty((8, 64), dtype=torch.float8_e4m3fn)
        b = torch.empty((64, 64), dtype=torch.float8_e4m3fn)
        a_s = torch.empty((8, 2), dtype=torch.float32)
        b_s = torch.empty((2, 2), dtype=torch.float32)
        fast_out = torch.full((8, 64), 1.0, dtype=torch.bfloat16)
        fallback_out = torch.full((8, 64), 2.0, dtype=torch.bfloat16)

        def _fake_prepare(A, B, As, Bs, block_size, output_dtype, output=None):
            return A.shape[0], B.shape[0], A.shape[1], fallback_out

        class _DummyKernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: None

        # Default (no published exec namespace, H20): routes to fast path before
        # prepare_block_fp8_matmul_inputs and config lookup.
        with (
            patch.object(
                fp8_kernel,
                "prepare_block_fp8_matmul_inputs",
                side_effect=AssertionError("redundant prepare called on fast path"),
            ),
            patch.object(fp8_kernel, "use_fp8_block32_sm90_gemm", return_value=True),
            patch.object(fp8_kernel, "is_h20_sm90", return_value=True),
            patch.object(
                fp8_kernel, "_fp8_block32_sm90_gemm_unchecked", return_value=fast_out
            ),
            patch.object(
                fp8_kernel,
                "get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: False
                ),
            ),
        ):
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
            )
            self.assertIs(out, fast_out)

        # Non-H20 SM90 (e.g. H200) keeps existing per-shape tuned JSON configs,
        # and uses the fast path when no tuned config exists.
        tuned_cfg = {
            16: {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
        }
        with (
            patch.object(fp8_kernel, "prepare_block_fp8_matmul_inputs", _fake_prepare),
            patch.object(fp8_kernel, "use_fp8_block32_sm90_gemm", return_value=True),
            patch.object(fp8_kernel, "is_h20_sm90", return_value=False),
            patch.object(
                fp8_kernel,
                "_fp8_block32_sm90_gemm_unchecked",
                side_effect=AssertionError("overrode tuned H200 JSON config"),
            ),
            patch.object(
                fp8_kernel, "get_w8a8_block_fp8_configs", return_value=tuned_cfg
            ),
            patch.object(
                fp8_kernel,
                "get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: False
                ),
            ),
            patch.object(
                fp8_kernel, "get_platform", return_value=SimpleNamespace(is_sm90=True)
            ),
            patch.object(fp8_kernel, "_w8a8_block_fp8_matmul", _DummyKernel()),
        ):
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
            )
            self.assertIs(out, fallback_out)

        with (
            patch.object(
                fp8_kernel,
                "prepare_block_fp8_matmul_inputs",
                side_effect=AssertionError("unseen H200 shape should use fast path"),
            ),
            patch.object(fp8_kernel, "use_fp8_block32_sm90_gemm", return_value=True),
            patch.object(fp8_kernel, "is_h20_sm90", return_value=False),
            patch.object(fp8_kernel, "get_w8a8_block_fp8_configs", return_value=None),
            patch.object(
                fp8_kernel, "_fp8_block32_sm90_gemm_unchecked", return_value=fast_out
            ),
            patch.object(
                fp8_kernel,
                "get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: False
                ),
            ),
        ):
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
            )
            self.assertIs(out, fast_out)

        # Deterministic inference must stay on the single-accumulator fallback.
        with (
            patch.object(fp8_kernel, "prepare_block_fp8_matmul_inputs", _fake_prepare),
            patch.object(fp8_kernel, "use_fp8_block32_sm90_gemm", return_value=True),
            patch.object(fp8_kernel, "is_h20_sm90", return_value=True),
            patch.object(
                fp8_kernel,
                "_fp8_block32_sm90_gemm_unchecked",
                side_effect=AssertionError("fast path ran in deterministic mode"),
            ),
            patch.object(
                fp8_kernel,
                "get_context",
                return_value=SimpleNamespace(
                    is_config_namespace_published=lambda name: name == "exec"
                ),
            ),
            patch.object(
                fp8_kernel,
                "get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(enable_deterministic_inference=True)
                ),
            ),
            patch.object(fp8_kernel, "get_w8a8_block_fp8_configs", return_value=None),
            patch.object(
                fp8_kernel, "get_platform", return_value=SimpleNamespace(is_sm90=True)
            ),
            patch.object(fp8_kernel, "_w8a8_block_fp8_matmul", _DummyKernel()),
        ):
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
            )
            self.assertIs(out, fallback_out)

    def test_is_h20_sm90_and_deepgemm_hybrid_preference(self):
        for arch, expected in (
            (_cuda_arch(9, 0, 78, "NVIDIA H20"), True),
            (_cuda_arch(9, 0, 78, "NVIDIA H20-3e"), True),
            (_cuda_arch(9, 0, 78, "NVIDIA H20 96GB"), True),
            (_cuda_arch(9, 0, 132, "NVIDIA H200"), False),
            (_cuda_arch(9, 0, 132, "NVIDIA H100 80GB HBM3"), False),
            (_cuda_arch(8, 0, 108, "NVIDIA A100"), False),
            (_hip_arch("gfx950"), False),
        ):
            with (
                self.subTest(arch=arch),
                patch.object(fp8_block32_sm90, "current_arch", return_value=arch),
            ):
                self.assertEqual(fp8_block32_sm90.is_h20_sm90(), expected)

        with patch.object(fp8_block32_sm90, "_sm_count", return_value=78):
            # M <= 16: narrow/medium N (< 0.75 * sm_count tiles) or short K (<= 320).
            self.assertTrue(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(1, 576, 5120)
            )
            self.assertTrue(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(16, 3200, 5120)
            )
            self.assertTrue(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(16, 5120, 288)
            )
            self.assertFalse(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(16, 4096, 4096)
            )
            self.assertFalse(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(16, 5120, 1024)
            )
            # 17 <= M <= 64: very narrow N (<= sm_count // 4 tiles).
            self.assertTrue(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(64, 576, 5120)
            )
            self.assertFalse(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(64, 1856, 5120)
            )
            # M > 64 and M == 0 stay on DeepGEMM.
            self.assertFalse(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(128, 576, 5120)
            )
            self.assertFalse(
                fp8_block32_sm90.prefers_fp8_block32_sm90_over_deepgemm(0, 576, 5120)
            )


if __name__ == "__main__":
    unittest.main()
