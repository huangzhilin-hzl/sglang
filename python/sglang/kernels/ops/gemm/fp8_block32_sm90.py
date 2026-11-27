"""SM90 (Hopper) Triton GEMM family for block-32x32 FP8 operands with FP32 scales.

Computes ``out[M, N] (bf16) = (A * As) @ (B * Bs)^T`` where:
  - ``A`` is ``[M, K]`` ``float8_e4m3fn`` with ``A.stride(1) == 1`` and ``A.stride(0) % 32 == 0``
  - ``B`` is contiguous ``[N, K]`` ``float8_e4m3fn``
  - ``As`` is ``[M, K // 32]`` ``float32`` with ``As.stride(1) == 1`` (or ``K == 32``)
  - ``Bs`` is ``[N // 32, K // 32]`` ``float32`` with ``Bs.stride(1) == 1`` (or ``K == 32``)
  - ``K > 0`` multiple of 32, ``N > 0`` multiple of 64, ``output_dtype == torch.bfloat16``

Rejects non-SM90 devices, non-``[32, 32]`` block sizes, non-bf16 outputs,
unaligned ``N``/``K``, and non-contiguous ``B``. Callers requiring batch-invariant
or deterministic execution must stay on the single-accumulator fallback because
this module selects distinct kernels and split-K reduction orders by ``M``.

Runtime dispatch splits across three ``M`` regimes using ``current_arch().num_sms``
so wave sizing adapts across SM90 parts (e.g. 78-SM H20/H20-3e, 114-SM
H100 PCIe, 132-SM H100/H200 SXM):
  1. ``M == 1`` with wide ``N`` and short ``K`` (``N // 32 >= 1.2 * sm_count`` and
     ``K // 32 <= 32``): single-pass direct-store GEMV (``BLOCK_N=32``) with
     cache hints (avoids split-K reduction overhead on short-K wide projections).
  2. ``1 <= M <= 16`` (all other ``M == 1`` shapes and small decode batches):
     Swap-AB Hopper WGMMA (``B @ A^T``, ``BLOCK_N=64``, ``BLOCK_M=16``), 2x
     K-group unrolled, with dynamic ``sm_count`` Split-K and zero-mask
     ``EXACT_EVEN_K`` / ``FULL_M`` specializations.
  3. ``M > 16`` (batched decode, speculative verify, and prefill): standard
     Hopper WGMMA (``BLOCK_M=64``, ``BLOCK_N in {128, 64}``) with
     ``GROUP_SIZE_M=32`` L2 swizzling and automatic Split-K when the ``(M, N)``
     tile grid is smaller than ``sm_count``.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.srt.utils.custom_op import register_custom_op

_SPLIT_K_CANDIDATES = (2, 3, 4, 5, 8, 16)
_H20_NAME = re.compile(r"NVIDIA H20($|[-\s])")


@triton.jit
def _fp8_block32_gemv_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    K: tl.constexpr,
    NUM_K_GROUPS: tl.constexpr,
    stride_b_scale_n,
    BLOCK_N: tl.constexpr,
):
    """Single-pass direct-store GEMV for M=1 wide-N, short-K projections."""
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 32)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    b_row_ptrs = b_ptr + offs_n.to(tl.int64)[:, None] * K
    b_scale_row_ptr = b_scale_ptr + pid_n.to(tl.int64) * stride_b_scale_n

    for k_group in range(0, NUM_K_GROUPS):
        offs_k_group = k_group * 32 + offs_k
        a_tile = tl.load(a_ptr + offs_k_group, eviction_policy="evict_last").to(
            tl.float32
        )
        b_tile = tl.load(
            b_row_ptrs + offs_k_group[None, :], eviction_policy="evict_first"
        ).to(tl.float32)
        a_scale = tl.load(a_scale_ptr + k_group)
        b_scale = tl.load(b_scale_row_ptr + k_group)
        accumulator += tl.sum(b_tile * a_tile[None, :], axis=1) * a_scale * b_scale

    tl.store(out_ptr + offs_n, accumulator.to(tl.bfloat16))


@triton.jit
def _fp8_block32_swapab64_wgmma_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    M,
    stride_split_k,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_GROUPS: tl.constexpr,
    stride_a_m,
    stride_a_scale_m,
    stride_b_scale_n,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
    EXACT_EVEN_K: tl.constexpr,
    FULL_M: tl.constexpr,
):
    """Hopper WGMMA Swap-AB (B @ A^T) with 64-row weight tiles and 2x K unroll."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * 64 + tl.arange(0, 64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 32)
    if not FULL_M:
        m_mask = offs_m < M

    stride_a_m_aligned = tl.multiple_of(stride_a_m, 32)
    offs_m_int64 = offs_m.to(tl.int64)
    a_row_ptrs = a_ptr + offs_m_int64[:, None] * stride_a_m_aligned
    b_row_ptrs = b_ptr + offs_n.to(tl.int64)[:, None] * K
    a_scale_row_ptrs = a_scale_ptr + offs_m_int64 * stride_a_scale_m

    # Each 64-row B tile spans two 32-row block-scale groups (lower and upper half).
    n_scale_group_base = pid_n.to(tl.int64) * 2
    b_scale_lower_row_ptr = b_scale_ptr + n_scale_group_base * stride_b_scale_n
    b_scale_upper_row_ptr = b_scale_ptr + (n_scale_group_base + 1) * stride_b_scale_n
    offs_n_local = tl.arange(0, 64)
    is_lower_n_half = offs_n_local[:, None] < 32

    accumulator = tl.zeros((64, BLOCK_M), dtype=tl.float32)

    if EXACT_EVEN_K:
        K_GROUPS_PER_SPLIT: tl.constexpr = NUM_K_GROUPS // SPLIT_K
        k_group_start = pid_k * K_GROUPS_PER_SPLIT
        for pair_idx in range(0, K_GROUPS_PER_SPLIT // 2):
            k_group_0 = k_group_start + pair_idx * 2
            offs_k_0 = tl.multiple_of(k_group_0 * 32, 32) + offs_k
            b_tile_0 = tl.load(b_row_ptrs + offs_k_0[None, :])
            if FULL_M:
                a_tile_0 = tl.load(a_row_ptrs + offs_k_0[None, :])
                a_scale_0 = tl.load(a_scale_row_ptrs + k_group_0)
            else:
                a_tile_0 = tl.load(
                    a_row_ptrs + offs_k_0[None, :], mask=m_mask[:, None], other=0.0
                )
                a_scale_0 = tl.load(
                    a_scale_row_ptrs + k_group_0, mask=m_mask, other=0.0
                )
            dot_0 = tl.dot(b_tile_0, tl.trans(a_tile_0))
            b_scale_0_lower = tl.load(b_scale_lower_row_ptr + k_group_0)
            b_scale_0_upper = tl.load(b_scale_upper_row_ptr + k_group_0)

            k_group_1 = k_group_0 + 1
            offs_k_1 = tl.multiple_of(k_group_1 * 32, 32) + offs_k
            b_tile_1 = tl.load(b_row_ptrs + offs_k_1[None, :])
            if FULL_M:
                a_tile_1 = tl.load(a_row_ptrs + offs_k_1[None, :])
                a_scale_1 = tl.load(a_scale_row_ptrs + k_group_1)
            else:
                a_tile_1 = tl.load(
                    a_row_ptrs + offs_k_1[None, :], mask=m_mask[:, None], other=0.0
                )
                a_scale_1 = tl.load(
                    a_scale_row_ptrs + k_group_1, mask=m_mask, other=0.0
                )
            dot_1 = tl.dot(b_tile_1, tl.trans(a_tile_1))
            b_scale_1_lower = tl.load(b_scale_lower_row_ptr + k_group_1)
            b_scale_1_upper = tl.load(b_scale_upper_row_ptr + k_group_1)

            # Combine scales before multiplying the 2D accumulator (matching
            # DeepGEMM SM90) so each block-32 promotion stays a single FFMA.
            scale_0_lower = b_scale_0_lower * a_scale_0
            scale_0_upper = b_scale_0_upper * a_scale_0
            accumulator += dot_0 * tl.where(
                is_lower_n_half, scale_0_lower[None, :], scale_0_upper[None, :]
            )
            scale_1_lower = b_scale_1_lower * a_scale_1
            scale_1_upper = b_scale_1_upper * a_scale_1
            accumulator += dot_1 * tl.where(
                is_lower_n_half, scale_1_lower[None, :], scale_1_upper[None, :]
            )
    else:
        k_groups_per_split = (NUM_K_GROUPS + SPLIT_K - 1) // SPLIT_K
        k_group_start = pid_k * k_groups_per_split
        k_group_end = tl.minimum(k_group_start + k_groups_per_split, NUM_K_GROUPS)
        k_group_count = tl.maximum(k_group_end - k_group_start, 0)
        k_group_pairs_end = k_group_start + (k_group_count // 2) * 2
        for k_group_0 in range(k_group_start, k_group_pairs_end, 2):
            offs_k_0 = tl.multiple_of(k_group_0 * 32, 32) + offs_k
            b_tile_0 = tl.load(b_row_ptrs + offs_k_0[None, :])
            if FULL_M:
                a_tile_0 = tl.load(a_row_ptrs + offs_k_0[None, :])
                a_scale_0 = tl.load(a_scale_row_ptrs + k_group_0)
            else:
                a_tile_0 = tl.load(
                    a_row_ptrs + offs_k_0[None, :], mask=m_mask[:, None], other=0.0
                )
                a_scale_0 = tl.load(
                    a_scale_row_ptrs + k_group_0, mask=m_mask, other=0.0
                )
            dot_0 = tl.dot(b_tile_0, tl.trans(a_tile_0))
            b_scale_0_lower = tl.load(b_scale_lower_row_ptr + k_group_0)
            b_scale_0_upper = tl.load(b_scale_upper_row_ptr + k_group_0)

            k_group_1 = k_group_0 + 1
            offs_k_1 = tl.multiple_of(k_group_1 * 32, 32) + offs_k
            b_tile_1 = tl.load(b_row_ptrs + offs_k_1[None, :])
            if FULL_M:
                a_tile_1 = tl.load(a_row_ptrs + offs_k_1[None, :])
                a_scale_1 = tl.load(a_scale_row_ptrs + k_group_1)
            else:
                a_tile_1 = tl.load(
                    a_row_ptrs + offs_k_1[None, :], mask=m_mask[:, None], other=0.0
                )
                a_scale_1 = tl.load(
                    a_scale_row_ptrs + k_group_1, mask=m_mask, other=0.0
                )
            dot_1 = tl.dot(b_tile_1, tl.trans(a_tile_1))
            b_scale_1_lower = tl.load(b_scale_lower_row_ptr + k_group_1)
            b_scale_1_upper = tl.load(b_scale_upper_row_ptr + k_group_1)

            # Combine scales before multiplying the 2D accumulator (matching
            # DeepGEMM SM90) so each block-32 promotion stays a single FFMA.
            scale_0_lower = b_scale_0_lower * a_scale_0
            scale_0_upper = b_scale_0_upper * a_scale_0
            accumulator += dot_0 * tl.where(
                is_lower_n_half, scale_0_lower[None, :], scale_0_upper[None, :]
            )
            scale_1_lower = b_scale_1_lower * a_scale_1
            scale_1_upper = b_scale_1_upper * a_scale_1
            accumulator += dot_1 * tl.where(
                is_lower_n_half, scale_1_lower[None, :], scale_1_upper[None, :]
            )

        if k_group_pairs_end < k_group_end:
            k_group_tail = k_group_pairs_end
            offs_k_tail = k_group_tail * 32 + offs_k
            b_tile_tail = tl.load(b_row_ptrs + offs_k_tail[None, :])
            if FULL_M:
                a_tile_tail = tl.load(a_row_ptrs + offs_k_tail[None, :])
                a_scale_tail = tl.load(a_scale_row_ptrs + k_group_tail)
            else:
                a_tile_tail = tl.load(
                    a_row_ptrs + offs_k_tail[None, :], mask=m_mask[:, None], other=0.0
                )
                a_scale_tail = tl.load(
                    a_scale_row_ptrs + k_group_tail, mask=m_mask, other=0.0
                )
            dot_tail = tl.dot(b_tile_tail, tl.trans(a_tile_tail))
            b_scale_tail_lower = tl.load(b_scale_lower_row_ptr + k_group_tail)
            b_scale_tail_upper = tl.load(b_scale_upper_row_ptr + k_group_tail)
            scale_tail_lower = b_scale_tail_lower * a_scale_tail
            scale_tail_upper = b_scale_tail_upper * a_scale_tail
            accumulator += dot_tail * tl.where(
                is_lower_n_half, scale_tail_lower[None, :], scale_tail_upper[None, :]
            )

    out_tile = tl.trans(accumulator)
    offs_out = offs_m_int64[:, None] * N + offs_n[None, :]
    if DIRECT_STORE:
        if FULL_M:
            tl.store(out_ptr + offs_out, out_tile.to(tl.bfloat16))
        else:
            tl.store(
                out_ptr + offs_out,
                out_tile.to(tl.bfloat16),
                mask=m_mask[:, None],
            )
    else:
        partial_base_ptr = out_ptr + pid_k.to(tl.int64) * tl.cast(
            stride_split_k, tl.int64
        )
        if FULL_M:
            tl.store(partial_base_ptr + offs_out, out_tile)
        else:
            tl.store(
                partial_base_ptr + offs_out,
                out_tile,
                mask=m_mask[:, None],
            )


@triton.jit
def _fp8_block32_splitk_reduce_kernel(
    partials_ptr,
    out_ptr,
    stride_split_k,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    partials_row_ptrs = partials_ptr + row_idx * N + offs_n
    stride_split_k_int64 = tl.cast(stride_split_k, tl.int64)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for split_idx in tl.static_range(SPLIT_K):
        accumulator += tl.load(
            partials_row_ptrs + split_idx * stride_split_k_int64,
            mask=n_mask,
            other=0.0,
        )
    tl.store(out_ptr + row_idx * N + offs_n, accumulator.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _fp8_block32_large_m_wgmma_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    out_ptr,
    M,
    stride_split_k,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_GROUPS: tl.constexpr,
    stride_a_m,
    stride_a_scale_m,
    stride_b_scale_n,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_a_m = offs_m % M
    if USE_INT64_OFFSETS:
        offs_a_m = offs_a_m.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 32)

    a_row_ptrs = a_ptr + offs_a_m[:, None] * stride_a_m
    b_row_ptrs = b_ptr + offs_n.to(tl.int64)[:, None] * K
    a_scale_row_ptrs = a_scale_ptr + offs_a_m * stride_a_scale_m

    offs_n_local = tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_groups_per_split = (NUM_K_GROUPS + SPLIT_K - 1) // SPLIT_K
    k_group_start = pid_k * k_groups_per_split
    k_group_end = tl.minimum(k_group_start + k_groups_per_split, NUM_K_GROUPS)

    if BLOCK_N == 128:
        n_scale_group_base = pid_n.to(tl.int64) * 4
        b_scale_row_ptr_0 = b_scale_ptr + (n_scale_group_base + 0) * stride_b_scale_n
        b_scale_row_ptr_1 = b_scale_ptr + (n_scale_group_base + 1) * stride_b_scale_n
        b_scale_row_ptr_2 = b_scale_ptr + (n_scale_group_base + 2) * stride_b_scale_n
        b_scale_row_ptr_3 = b_scale_ptr + (n_scale_group_base + 3) * stride_b_scale_n
        for k_group in range(k_group_start, k_group_end):
            offs_k_group = k_group * 32 + offs_k
            a_tile = tl.load(a_row_ptrs + offs_k_group[None, :])
            b_tile = tl.load(b_row_ptrs + offs_k_group[None, :])
            dot = tl.dot(a_tile, tl.trans(b_tile))
            a_scale = tl.load(a_scale_row_ptrs + k_group)
            b_scale_0 = tl.load(b_scale_row_ptr_0 + k_group)
            b_scale_1 = tl.load(b_scale_row_ptr_1 + k_group)
            b_scale_2 = tl.load(b_scale_row_ptr_2 + k_group)
            b_scale_3 = tl.load(b_scale_row_ptr_3 + k_group)
            # Combine scales before multiplying the 2D accumulator (matching
            # DeepGEMM SM90) so each block-32 promotion stays a single FFMA.
            scale_lower_half = tl.where(
                offs_n_local[None, :] < 32,
                (a_scale * b_scale_0)[:, None],
                (a_scale * b_scale_1)[:, None],
            )
            scale_upper_half = tl.where(
                offs_n_local[None, :] < 96,
                (a_scale * b_scale_2)[:, None],
                (a_scale * b_scale_3)[:, None],
            )
            accumulator += dot * tl.where(
                offs_n_local[None, :] < 64, scale_lower_half, scale_upper_half
            )
    else:
        n_scale_group_base = pid_n.to(tl.int64) * 2
        b_scale_row_ptr_0 = b_scale_ptr + (n_scale_group_base + 0) * stride_b_scale_n
        b_scale_row_ptr_1 = b_scale_ptr + (n_scale_group_base + 1) * stride_b_scale_n
        for k_group in range(k_group_start, k_group_end):
            offs_k_group = k_group * 32 + offs_k
            a_tile = tl.load(a_row_ptrs + offs_k_group[None, :])
            b_tile = tl.load(b_row_ptrs + offs_k_group[None, :])
            dot = tl.dot(a_tile, tl.trans(b_tile))
            a_scale = tl.load(a_scale_row_ptrs + k_group)
            b_scale_0 = tl.load(b_scale_row_ptr_0 + k_group)
            b_scale_1 = tl.load(b_scale_row_ptr_1 + k_group)
            accumulator += dot * tl.where(
                offs_n_local[None, :] < 32,
                (a_scale * b_scale_0)[:, None],
                (a_scale * b_scale_1)[:, None],
            )

    offs_out_m = offs_m.to(tl.int64) if USE_INT64_OFFSETS else offs_m
    m_mask = offs_m < M
    offs_out = offs_out_m[:, None] * N + offs_n[None, :]
    if DIRECT_STORE:
        tl.store(out_ptr + offs_out, accumulator.to(tl.bfloat16), mask=m_mask[:, None])
    else:
        partial_base_ptr = out_ptr + pid_k.to(tl.int64) * tl.cast(
            stride_split_k, tl.int64
        )
        tl.store(partial_base_ptr + offs_out, accumulator, mask=m_mask[:, None])


def _sm_count() -> int:
    return current_arch().num_sms


def is_h20_sm90() -> bool:
    """Whether the current GPU is an H20-family SM90 part (H20 / H20-3e, not H200)."""
    arch = current_arch()
    return arch.is_sm(9, 0) and bool(_H20_NAME.match(arch.name))


def prefers_fp8_block32_sm90_over_deepgemm(m: int, n: int, k: int) -> bool:
    """Whether the SM90 Triton block-32x32 kernel beats DeepGEMM on H20.

    Covers low-batch decode where DeepGEMM's persistent WGMMA (no split-K)
    underutilizes SMs or pays TMA setup over short K, and narrow-N medium-M
    projections where output tiles leave most SMs idle.
    """
    if m <= 0:
        return False
    sm_count = _sm_count()
    num_n_tiles = n // 64
    if m <= 16:
        return num_n_tiles < (sm_count * 3) // 4 or k <= 320
    if m <= 64:
        return num_n_tiles <= sm_count // 4
    return False


def _reduce_block_n(n: int) -> int:
    if n >= 4096:
        return 1024
    if n >= 2048:
        return 512
    if n >= 1024:
        return 256
    return 64


def _is_sm90_device(device: torch.device) -> bool:
    return torch.version.hip is None and torch.cuda.get_device_capability(device) == (
        9,
        0,
    )


def _needs_int64_offsets(
    A: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    m: int,
    n: int,
    k: int,
    num_k_groups: int,
    split_k: int,
    block_m: int,
) -> bool:
    """Return whether any element offset can reach or exceed 2**31."""
    max_a_offset = (m - 1) * A.stride(0) + (k - 1)
    max_b_offset = n * k - 1
    max_a_scale_offset = (m - 1) * As.stride(0) + (num_k_groups - 1)
    max_b_scale_offset = (n // 32 - 1) * Bs.stride(0) + (num_k_groups - 1)
    padded_m = triton.cdiv(m, block_m) * block_m
    max_out_offset = max(split_k * m * n, padded_m * n) - 1
    return (
        max(
            max_a_offset,
            max_b_offset,
            max_a_scale_offset,
            max_b_scale_offset,
            max_out_offset,
        )
        >= 2**31
    )


def _choose_split_k(num_n_tiles: int, num_k_groups: int, sm_count: int) -> int:
    """Pick a split-K factor targeting ~2 resident CTAs per SM.

    Uses the runtime ``sm_count`` so wave sizing adapts across SM90 GPUs
    (78-SM H20/H20-3e, 114-SM H100 PCIe, 132-SM H100/H200 SXM).
    """
    if num_k_groups <= 1 or sm_count <= 0:
        return 1
    # Once N-tiles already fill well past one SM wave, or fill one wave with
    # short K, splitting K introduces a tail wave plus a reduction launch.
    if num_n_tiles >= (sm_count * 6) // 5:
        return 1
    if num_n_tiles >= sm_count and num_k_groups <= 16:
        return 1

    target_ctas = 2 * sm_count
    min_k_groups_per_split = 8 if num_n_tiles >= max(1, sm_count // 4) else 4
    best_split_k = 1
    best_error = abs(num_n_tiles - target_ctas)
    for split_k in _SPLIT_K_CANDIDATES:
        if split_k > num_k_groups or num_k_groups // split_k < min_k_groups_per_split:
            continue
        error = abs(num_n_tiles * split_k - target_ctas)
        if error < best_error:
            best_error = error
            best_split_k = split_k
    return best_split_k


def use_fp8_block32_sm90_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: Sequence[int],
    output_dtype: torch.dtype,
) -> bool:
    """Return whether the SM90 block-32x32 FP8 Triton fast path applies."""
    if not current_arch().is_sm(9, 0):
        return False
    if (
        len(block_size) != 2
        or block_size[0] != 32
        or block_size[1] != 32
        or output_dtype != torch.bfloat16
    ):
        return False
    if not (
        A.is_cuda
        and B.is_cuda
        and As.is_cuda
        and Bs.is_cuda
        and A.device == B.device == As.device == Bs.device
        and A.dtype == torch.float8_e4m3fn
        and B.dtype == torch.float8_e4m3fn
        and As.dtype == torch.float32
        and Bs.dtype == torch.float32
        and A.ndim == 2
        and B.ndim == 2
        and As.ndim == 2
        and Bs.ndim == 2
    ):
        return False
    m, k = A.shape
    n, weight_k = B.shape
    return (
        k == weight_k
        and k > 0
        and k % 32 == 0
        and n > 0
        and n % 64 == 0
        and A.stride(1) == 1
        and A.stride(0) >= 0
        and A.stride(0) % 32 == 0
        and B.is_contiguous()
        and As.stride(0) >= 0
        and (As.stride(1) == 1 or As.shape[1] == 1)
        and Bs.stride(0) >= 0
        and (Bs.stride(1) == 1 or Bs.shape[1] == 1)
        and As.shape == (m, k // 32)
        and Bs.shape == (n // 32, k // 32)
    )


def _launch_fp8_block32_sm90_gemm(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> None:
    m, k = A.shape
    if m == 0:
        return
    n = B.shape[0]
    num_k_groups = k // 32
    sm_count = _sm_count()

    if m == 1 and (n // 32) >= (sm_count * 6) // 5 and num_k_groups <= 32:
        _fp8_block32_gemv_kernel[(n // 32,)](
            A,
            B,
            As,
            Bs,
            out,
            K=k,
            NUM_K_GROUPS=num_k_groups,
            stride_b_scale_n=Bs.stride(0),
            BLOCK_N=32,
            num_warps=4,
            num_stages=3,
        )
    elif m <= 16:
        block_m = 16
        num_n_tiles = n // 64
        split_k = _choose_split_k(num_n_tiles, num_k_groups, sm_count)
        stride_split_k = m * n
        direct_store = split_k == 1
        target = (
            out
            if direct_store
            else torch.empty((split_k, m, n), dtype=torch.float32, device=A.device)
        )
        _fp8_block32_swapab64_wgmma_kernel[(num_n_tiles, split_k)](
            A,
            B,
            As,
            Bs,
            target,
            m,
            stride_split_k,
            N=n,
            K=k,
            NUM_K_GROUPS=num_k_groups,
            stride_a_m=A.stride(0),
            stride_a_scale_m=As.stride(0),
            stride_b_scale_n=Bs.stride(0),
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            DIRECT_STORE=direct_store,
            EXACT_EVEN_K=(num_k_groups % (split_k * 2) == 0),
            FULL_M=(m == block_m),
            num_warps=4,
            num_stages=4,
        )
        if not direct_store:
            reduce_block_n = _reduce_block_n(n)
            _fp8_block32_splitk_reduce_kernel[(m, triton.cdiv(n, reduce_block_n))](
                target,
                out,
                stride_split_k,
                N=n,
                SPLIT_K=split_k,
                BLOCK_N=reduce_block_n,
                num_warps=4 if reduce_block_n >= 256 else 2,
            )
    else:
        block_m = 64
        num_pid_m = triton.cdiv(m, block_m)
        use_block_n_128 = n % 128 == 0 and (
            n >= 1024 or num_pid_m * (n // 128) >= sm_count
        )
        block_n = 128 if use_block_n_128 else 64
        num_tiles = num_pid_m * (n // block_n)
        split_k = (
            _choose_split_k(num_tiles, num_k_groups, sm_count)
            if num_tiles < sm_count
            else 1
        )
        stride_split_k = m * n
        use_int64_offsets = _needs_int64_offsets(
            A, As, Bs, m, n, k, num_k_groups, split_k, block_m
        )
        direct_store = split_k == 1
        target = (
            out
            if direct_store
            else torch.empty((split_k, m, n), dtype=torch.float32, device=A.device)
        )
        _fp8_block32_large_m_wgmma_kernel[(num_tiles, split_k)](
            A,
            B,
            As,
            Bs,
            target,
            m,
            stride_split_k,
            N=n,
            K=k,
            NUM_K_GROUPS=num_k_groups,
            stride_a_m=A.stride(0),
            stride_a_scale_m=As.stride(0),
            stride_b_scale_n=Bs.stride(0),
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            GROUP_SIZE_M=32,
            DIRECT_STORE=direct_store,
            USE_INT64_OFFSETS=use_int64_offsets,
            num_warps=4,
            num_stages=3,
        )
        if not direct_store:
            reduce_block_n = _reduce_block_n(n)
            _fp8_block32_splitk_reduce_kernel[(m, triton.cdiv(n, reduce_block_n))](
                target,
                out,
                stride_split_k,
                N=n,
                SPLIT_K=split_k,
                BLOCK_N=reduce_block_n,
                num_warps=4 if reduce_block_n >= 256 else 2,
            )


@register_custom_op(op_name="fp8_block32_sm90_gemm", mutates_args=["out"])
def _fp8_block32_sm90_gemm_custom_op(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> None:
    _launch_fp8_block32_sm90_gemm(out, A, B, As, Bs)


def _fp8_block32_sm90_gemm_unchecked(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast dispatch path after eligibility has already been validated."""
    if out is None:
        out = A.new_empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16)
    _fp8_block32_sm90_gemm_custom_op(out, A, B, As, Bs)
    return out


def fp8_block32_sm90_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SM90 Triton GEMM for block-32x32 FP8 operands with FP32 scales -> BF16."""
    if A.ndim != 2 or B.ndim != 2 or As.ndim != 2 or Bs.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors, got A={A.shape}, B={B.shape}, "
            f"As={As.shape}, Bs={Bs.shape}"
        )
    m, k = A.shape
    n, weight_k = B.shape
    if k != weight_k or k == 0 or k % 32 != 0 or n == 0 or n % 64 != 0:
        raise ValueError(
            f"Unsupported shape for block-32x32 SM90 FP8 GEMM: A={A.shape}, "
            f"B={B.shape} (require K>0 multiple of 32 and N>0 multiple of 64)"
        )
    if As.shape != (m, k // 32) or Bs.shape != (n // 32, k // 32):
        raise ValueError(
            f"Scale shape mismatch: As={As.shape}, Bs={Bs.shape} "
            f"for A={A.shape}, B={B.shape}"
        )
    if not (
        A.dtype == torch.float8_e4m3fn
        and B.dtype == torch.float8_e4m3fn
        and As.dtype == torch.float32
        and Bs.dtype == torch.float32
        and A.is_cuda
        and A.device == B.device == As.device == Bs.device
    ):
        raise ValueError(
            f"Expected same-device CUDA fp8_e4m3fn inputs and fp32 scales, got "
            f"A=({A.dtype}, {A.device}), B=({B.dtype}, {B.device}), "
            f"As=({As.dtype}, {As.device}), Bs=({Bs.dtype}, {Bs.device})"
        )
    if not _is_sm90_device(A.device):
        raise ValueError(
            f"fp8_block32_sm90_gemm requires an NVIDIA SM90 CUDA device, got {A.device}"
        )
    if not (
        A.stride(1) == 1
        and A.stride(0) >= 0
        and A.stride(0) % 32 == 0
        and B.is_contiguous()
        and As.stride(0) >= 0
        and (As.stride(1) == 1 or As.shape[1] == 1)
        and Bs.stride(0) >= 0
        and (Bs.stride(1) == 1 or Bs.shape[1] == 1)
    ):
        raise ValueError(
            f"Unsupported tensor strides for block-32x32 SM90 FP8 GEMM: "
            f"A.stride={A.stride()}, B.stride={B.stride()}, "
            f"As.stride={As.stride()}, Bs.stride={Bs.stride()}"
        )

    if out is None:
        out = A.new_empty((m, n), dtype=torch.bfloat16)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"Expected contiguous bf16 out of shape {(m, n)} on {A.device}, "
            f"got shape={out.shape}, dtype={out.dtype}, device={out.device}, "
            f"contiguous={out.is_contiguous()}"
        )

    _fp8_block32_sm90_gemm_custom_op(out, A, B, As, Bs)
    return out
