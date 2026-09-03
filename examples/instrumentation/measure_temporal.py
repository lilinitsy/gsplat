#!/usr/bin/env python
"""Phase 0 / WP-C: temporal visibility statistics (B1-B5, C1).

For pose pairs (A, B) derived from base views of a trained 2DGS scene, measure:

* B1  per-pixel contributing-set IoU between A and B (correspondence via B's
      median depth),
* B2  "oracle warp": render B from A's per-pixel candidate sets only
      (fresh transforms / order / SH) -> PSNR / LPIPS vs. B's full render,
* B3  fallback fraction (oracle alpha < full alpha - tau),
* B4  order inversions among shared contributors + stale-order composite,
* B5  candidate union size / N and 2x2-quad ID sharing,
* C1  oracle composite with A's stale per-surfel RGB (no SH re-eval).

Outputs ``<out>/temporal_pairs.csv`` (exact columns from PLAN.md),
``<out>/temporal_pairs_extra.csv`` (auxiliary stats: cycle-consistent IoU,
timing) and PNGs under ``<out>/png/``.

Conventions
-----------
* ``viewmat`` is world-to-camera, OpenCV (+z forward), as everywhere in gsplat.
* Pixel ``p = i * W + j`` (row-major); pixel centre ``(j + 0.5, i + 0.5)``.
* Contributing sets are the kernel's contributors with blend weight ``w > eps``.
* Correspondence B->A uses B's median depth. A B pixel *has a correspondence*
  when it has any contributor (median depth > 0) and projects inside A with
  ``z_A > 0``; it is *valid* (used for IoU / inversion statistics and
  ``psnr_oracle_valid``) when additionally ``acc_alpha_B > 0.5``.  The oracle
  warp is attempted for every pixel with a correspondence (a real warp renderer
  has no better guess for semi-transparent pixels); ``--strict_valid`` restricts
  it to valid pixels only.
* Candidate cap policy (``--cap_policy``): ``center_depth`` (default) keeps the
  members of S_A(q(p)) first, then neighbourhood-only surfels by A-depth;
  ``depth`` keeps the K nearest by A-depth regardless of origin (PLAN wording).
  ``cap_bind_frac`` reports how often the union exceeded K either way.
* ``--selfcheck`` adds two identity pairs (A == B): ``identity_exact`` runs with
  eps = 0 and an uncapped K and must reproduce B's render (IoU 1, PSNR >= 50 dB,
  no fallback, no inversions) -- this isolates pipeline correctness from the two
  deliberate approximations; ``identity`` runs at the requested eps / K and
  documents the truncation floor at delta = 0.

Run from ``examples/``::

    python instrumentation/measure_temporal.py --ckpt results/garden_2dgs/ckpts/ckpt_29999.pt \
        --data_dir data/360_v2/garden --out instrumentation/out
    python instrumentation/measure_temporal.py --synthetic --selfcheck --out instrumentation/out_synth
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    composite,
    eval_alpha,
    extract_contributors,
    load_cameras,
    load_splats,
    lpips_fn,
    make_synthetic_scene,
    psnr,
    render_stages,
    save_png,
    scale_K,
    segment_weights,
)

ROT_DELTAS_DEG: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
TRANS_DELTAS_PCT: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_TRAJ_PAIRS = 5
KINDS_ALL = ["rot_yaw", "rot_pitch", "trans_x", "trans_z", "traj"]

CSV_COLUMNS_FIXED = [
    "view", "kind", "delta", "rot_deg", "trans_frac", "N", "eps", "K", "radius",
    "valid_frac", "iou_mean", "iou_p10", "iou_p50", "cap_bind_frac",
    "psnr_oracle", "lpips_oracle", "psnr_oracle_valid",
]
CSV_COLUMNS_TAIL = [
    "inversion_frac", "inv_depth_gap_p50", "psnr_stale_order",
    "union_frac", "quad_share", "psnr_stale_sh", "mean_candidates_per_pixel",
]
EXTRA_COLUMNS = [
    "view", "kind", "delta", "has_corr_frac", "valid_cycle_frac", "iou_mean_cycle",
    "iou_p10_cycle", "iou_p50_cycle", "psnr_stale_order_valid", "psnr_stale_sh_valid",
    "lpips_oracle_valid_masked", "n_candidate_pairs", "n_shared_pairs", "cap_policy", "time_s",
]

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Small vectorised helpers
# --------------------------------------------------------------------------- #
def tau_col(t: float) -> str:
    """Column name for a fallback threshold, e.g. 0.01 -> 'fallback_frac_tau0.01'."""
    s = f"{t:g}"
    return f"fallback_frac_tau{s}"


def fmt(x) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.6g}"
    return str(x)


def lexsort_by_pixel(pix: Tensor, val: Tensor) -> Tensor:
    """Indices sorting by ``(pix, val)`` lexicographically (ascending)."""
    idx = torch.argsort(val)
    idx = idx[torch.argsort(pix[idx], stable=True)]
    return idx


def gather_segments(offsets: Tensor, gs: Tensor, q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Concatenate the CSR segments ``gs[offsets[q_s]:offsets[q_s+1]]`` for each slot s.

    Returns ``(slot [T], g [T], within [T])`` where ``within`` is the 0-based
    position inside the segment (i.e. the front-to-back rank).
    """
    device = gs.device
    starts = offsets[q]
    lens = offsets[q + 1] - starts
    total = int(lens.sum().item())
    slot = torch.repeat_interleave(torch.arange(q.numel(), device=device, dtype=torch.int64), lens)
    excl = torch.cumsum(lens, dim=0) - lens
    within = torch.arange(total, device=device, dtype=torch.int64) - excl[slot]
    return slot, gs[starts[slot] + within], within


def csr_offsets(pix_local: Tensor, n_pix: int) -> Tuple[Tensor, Tensor]:
    """CSR offsets for entries grouped by ``pix_local`` (must be non-decreasing)."""
    cnt = torch.bincount(pix_local, minlength=n_pix)
    offsets = torch.zeros(n_pix + 1, dtype=torch.int64, device=pix_local.device)
    offsets[1:] = torch.cumsum(cnt, dim=0)
    return offsets, cnt


def prune_stage(st: dict) -> dict:
    """Drop everything not needed after contributor extraction (memory)."""
    keep = (
        "radii", "means2d", "depths", "ray_transforms", "opacities", "colors_rgb",
        "render_colors", "render_alphas", "render_median", "viewmat", "K",
        "W", "H", "N", "tile_size",
    )
    return {k: st[k] for k in keep}


@torch.no_grad()
def contributing_sets(st: dict, W: int, H: int, eps: float) -> dict:
    """Kernel contributors with weight ``w > eps``, grouped by pixel, front-to-back.

    Returns ``gs [M]``, ``offsets [H*W+1]``, ``counts [H*W]``, ``M``.
    """
    con = extract_contributors(st, W, H)
    gs, pix, offsets = con["gs_ids"], con["pix_ids"], con["offsets"]
    alpha = eval_alpha(st, gs, pix, W)
    _, w = segment_weights(alpha, offsets)
    keep = w > eps
    gs = gs[keep]
    pix = pix[keep]
    del con, alpha, w, keep
    offsets, cnt = csr_offsets(pix, H * W)
    return {"gs": gs, "offsets": offsets, "counts": cnt, "M": int(gs.numel())}


@torch.no_grad()
def correspond(
    st_src: dict, c2w_src: Tensor, viewmat_dst: Tensor, K_dst: Tensor, W: int, H: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map every source pixel (via its median depth) to a destination pixel.

    Returns ``(q [P] int64 (dst pixel id, 0 where !has), has [P] bool,
    qi [P], qj [P])``.  ``has`` = median depth > 0, z_dst > 0, inside image.
    """
    device = st_src["means2d"].device
    P = H * W
    depth = st_src["render_median"][0, ..., 0].reshape(-1).to(torch.float64)
    Ks = st_src["K"].to(torch.float64)
    Kd = K_dst.to(torch.float64)
    c2w = c2w_src.to(torch.float64)
    vm = viewmat_dst.to(torch.float64)
    pix = torch.arange(P, device=device, dtype=torch.int64)
    ii = torch.div(pix, W, rounding_mode="floor").to(torch.float64)
    jj = (pix % W).to(torch.float64)
    x = (jj + 0.5 - Ks[0, 2]) / Ks[0, 0] * depth
    y = (ii + 0.5 - Ks[1, 2]) / Ks[1, 1] * depth
    pts = torch.stack([x, y, depth], dim=-1)  # camera-space (src)
    world = pts @ c2w[:3, :3].T + c2w[:3, 3]
    cam = world @ vm[:3, :3].T + vm[:3, 3]
    z = cam[:, 2]
    zs = torch.where(z > 0, z, torch.ones_like(z))
    u = Kd[0, 0] * cam[:, 0] / zs + Kd[0, 2]
    v = Kd[1, 1] * cam[:, 1] / zs + Kd[1, 2]
    qj = torch.round(u - 0.5).to(torch.int64)
    qi = torch.round(v - 0.5).to(torch.int64)
    has = (depth > 0) & (z > 0) & (qj >= 0) & (qj < W) & (qi >= 0) & (qi < H)
    q = torch.where(has, qi * W + qj, torch.zeros_like(qi))
    return q, has, qi, qj


# --------------------------------------------------------------------------- #
# Pose helpers
# --------------------------------------------------------------------------- #
def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def local_delta(c2w_A: np.ndarray, R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None) -> np.ndarray:
    D = np.eye(4, dtype=np.float64)
    if R is not None:
        D[:3, :3] = R
    if t is not None:
        D[:3, 3] = t
    return c2w_A.astype(np.float64) @ D


def pose_delta(c2w_A: np.ndarray, c2w_B: np.ndarray, scene_scale: float) -> Tuple[float, float]:
    """(geodesic rotation angle in degrees, |dt| / scene_scale)."""
    RA, RB = c2w_A[:3, :3], c2w_B[:3, :3]
    Rrel = RA.T @ RB
    cos = (np.trace(Rrel) - 1.0) / 2.0
    rot_deg = math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))
    trans = float(np.linalg.norm(c2w_B[:3, 3] - c2w_A[:3, 3])) / float(scene_scale)
    return rot_deg, trans


def trajectory_pairs(c2ws: np.ndarray, scene_scale: float, n_pairs: int, max_rot_deg: float = 1.0,
                     max_trans_frac: float = 0.02) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """Interpolated path through the base views; ``n_interp`` doubled until the
    first ``n_pairs`` consecutive steps are all below the one-frame thresholds."""
    from datasets.traj import generate_interpolated_path

    poses = np.asarray(c2ws, dtype=np.float64)[:, :3, :]
    n_interp = 1
    path4 = None
    while n_interp <= 1 << 14:
        path = generate_interpolated_path(poses, n_interp)  # [n_interp*(n-1),3,4]
        bottom = np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]]), (len(path), 1, 1))
        path4 = np.concatenate([path, bottom], axis=1)
        if len(path4) >= n_pairs + 1:
            ok = True
            for k in range(n_pairs):
                r, t = pose_delta(path4[k], path4[k + 1], scene_scale)
                if r >= max_rot_deg or t >= max_trans_frac:
                    ok = False
                    break
            if ok:
                break
        n_interp *= 2
    pairs = [(path4[k], path4[k + 1]) for k in range(min(n_pairs, len(path4) - 1))]
    return pairs, n_interp


# --------------------------------------------------------------------------- #
# The per-pair measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_pair(
    splats: dict,
    sh_degree: Optional[int],
    c2w_A: np.ndarray,
    c2w_B: np.ndarray,
    K_A: Tensor,
    K_B: Tensor,
    W: int,
    H: int,
    args: argparse.Namespace,
    lp: Optional[Callable],
    taus: List[float],
    eps: float,
    K_cap: int,
    png_prefix: Optional[str] = None,
    png_fallback_only: bool = False,
) -> Dict[str, float]:
    device = splats["means"].device
    t_start = time.perf_counter()
    P = H * W
    N = int(splats["N"])
    r = int(args.radius)

    c2w_A_t = torch.as_tensor(c2w_A, dtype=torch.float64, device=device)
    c2w_B_t = torch.as_tensor(c2w_B, dtype=torch.float64, device=device)
    vm_A = torch.inverse(c2w_A_t)
    vm_B = torch.inverse(c2w_B_t)

    # ---- 1. render both views, contributing sets ------------------------------
    stA = render_stages(splats, vm_A.float(), K_A, W, H, sh_degree=sh_degree)
    SA = contributing_sets(stA, W, H, eps)
    stA = prune_stage(stA)
    torch.cuda.empty_cache()
    stB = render_stages(splats, vm_B.float(), K_B, W, H, sh_degree=sh_degree)
    SB = contributing_sets(stB, W, H, eps)
    stB = prune_stage(stB)
    torch.cuda.empty_cache()

    depths_A = stA["depths"][0]  # [N]
    depths_B = stB["depths"][0]
    col_A = stA["colors_rgb"][0]  # [N,3]
    col_B = stB["colors_rgb"][0]
    full_rgb = stB["render_colors"][0]  # [H,W,3]
    full_alpha = stB["render_alphas"][0, ..., 0]  # [H,W]
    acc_A = stA["render_alphas"][0, ..., 0].reshape(-1)

    # ---- 2. correspondence -----------------------------------------------------
    q, has_corr, _, _ = correspond(stB, c2w_B_t, vm_A, stA["K"], W, H)
    valid = has_corr & (full_alpha.reshape(-1) > 0.5)
    # A->B back-projection for an occlusion-aware (cycle-consistent) pixel subset
    qb, has_b, qbi, qbj = correspond(stA, c2w_A_t, vm_B, stB["K"], W, H)
    pix_all = torch.arange(P, device=device, dtype=torch.int64)
    pi = torch.div(pix_all, W, rounding_mode="floor")
    pj = pix_all % W
    valid_cycle = (
        valid & has_b[q] & (acc_A[q] > 0.5)
        & ((qbi[q] - pi).abs() <= 1) & ((qbj[q] - pj).abs() <= 1)
    )
    if args.strict_valid:
        has_corr = valid

    # ---- accumulators ----------------------------------------------------------
    iou = torch.full((P,), NAN, dtype=torch.float32, device=device)
    oracle_rgb = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    oracle_alpha = torch.zeros((H, W), dtype=torch.float32, device=device)
    stale_order_rgb = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    stale_sh_rgb = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    union_mask = torch.zeros((N,), dtype=torch.bool, device=device)
    n_cap_bind = 0
    n_cand_total = 0
    n_cand_pixels = 0
    n_disc = 0
    n_pairs_total = 0
    n_shared_total = 0
    quad_num = 0
    quad_den = 0
    gap_chunks: List[Tensor] = []
    gap_cap = int(args.gap_samples)

    Kp_max = int(args.inv_pad)
    offs = torch.arange(-r, r + 1, device=device, dtype=torch.int64)
    n_nb = (2 * r + 1) ** 2
    chunk_rows = max(2, (int(args.chunk_px) // W) // 2 * 2)
    Wq = W // 2
    Hq2 = 2 * (H // 2)
    Wq2 = 2 * Wq

    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        p0, p1 = r0 * W, r1 * W
        Pc = p1 - p0
        pix_c = pix_all[p0:p1]

        # ---------------- B1: IoU + B4: inversions (valid pixels) ----------------
        pv = pix_c[valid[p0:p1]]
        nv = int(pv.numel())
        if nv > 0:
            qv = q[pv]
            slotA, gA, rankA = gather_segments(SA["offsets"], SA["gs"], qv)
            slotB, gB, rankB = gather_segments(SB["offsets"], SB["gs"], pv)
            keysA = pv[slotA] * N + gA
            keysB = pv[slotB] * N + gB
            matchB = torch.isin(keysB, keysA)
            nA = torch.bincount(slotA, minlength=nv)
            nB = torch.bincount(slotB, minlength=nv)
            inter = torch.bincount(slotB[matchB], minlength=nv)
            union = nA + nB - inter
            iou_v = torch.where(union > 0, inter.float() / union.clamp(min=1).float(), torch.ones_like(inter, dtype=torch.float32))
            iou[pv] = iou_v

            # inversions among shared surfels: ranks in A's list vs B's list
            if bool(matchB.any()):
                keysA_s, ordA = torch.sort(keysA)
                rankA_s = rankA[ordA]
                mk = keysB[matchB]
                pos = torch.searchsorted(keysA_s, mk)
                rA = rankA_s[pos]
                slot_m = slotB[matchB]  # non-decreasing; within a slot ordered by B rank
                gm = gB[matchB]
                m_cnt = torch.bincount(slot_m, minlength=nv)
                Kp = int(min(int(m_cnt.max().item()), Kp_max))
                within_m = torch.arange(slot_m.numel(), device=device) - (torch.cumsum(m_cnt, 0) - m_cnt)[slot_m]
                keep_m = within_m < Kp
                n_shared_total += int(slot_m.numel())
                RA = torch.full((nv, Kp), -1, dtype=torch.int32, device=device)
                DA = torch.zeros((nv, Kp), dtype=torch.float32, device=device)
                RA[slot_m[keep_m], within_m[keep_m]] = rA[keep_m].to(torch.int32)
                DA[slot_m[keep_m], within_m[keep_m]] = depths_A[gm[keep_m]]
                tri = torch.triu(torch.ones((Kp, Kp), dtype=torch.bool, device=device), diagonal=1)
                sub = int(args.inv_sub)
                for s0 in range(0, nv, sub):
                    ra = RA[s0:s0 + sub]
                    da = DA[s0:s0 + sub]
                    okm = ra >= 0
                    mc = okm.sum(dim=1).to(torch.int64)
                    n_pairs_total += int((mc * (mc - 1) // 2).sum().item())
                    both = okm[:, :, None] & okm[:, None, :] & tri[None]
                    disc = both & (ra[:, :, None] > ra[:, None, :])
                    nd = int(disc.sum().item())
                    if nd > 0:
                        n_disc += nd
                        gaps = (da[:, :, None] - da[:, None, :]).abs()[disc]
                        if gaps.numel() > gap_cap:
                            sel = torch.randint(0, gaps.numel(), (gap_cap,), device=device)
                            gaps = gaps[sel]
                        gap_chunks.append(gaps)
                    del both, disc
                del RA, DA
            del slotA, gA, rankA, slotB, gB, rankB, keysA, keysB, matchB

        # ---------------- B2/B3/B4/B5/C1: candidate sets + oracle ------------------
        ph = pix_c[has_corr[p0:p1]]
        nh = int(ph.numel())
        n_cand_pixels += nh
        if nh == 0:
            continue
        qh = q[ph]
        qi = torch.div(qh, W, rounding_mode="floor")
        qj = qh % W
        nbi = (qi[:, None] + offs[None, :]).clamp(0, H - 1)  # [nh, 2r+1]
        nbj = (qj[:, None] + offs[None, :]).clamp(0, W - 1)
        nb = (nbi[:, :, None] * W + nbj[:, None, :]).reshape(-1)  # [nh * n_nb]
        owner = torch.repeat_interleave(torch.arange(nh, device=device, dtype=torch.int64), n_nb)
        slotN, gN, _ = gather_segments(SA["offsets"], SA["gs"], nb)
        keys = ph[owner[slotN]] * N + gN
        del slotN, gN, nb, owner, nbi, nbj
        keys_u = torch.unique(keys)  # sorted -> grouped by pixel ascending
        del keys
        pc = torch.div(keys_u, N, rounding_mode="floor")
        gc = keys_u % N
        # membership / rank in the centre pixel's own list S_A(q(p))
        slotC, gC, rankC = gather_segments(SA["offsets"], SA["gs"], qh)
        keysC = ph[slotC] * N + gC
        if keysC.numel() > 0:
            keysC_s, ordC = torch.sort(keysC)
            rankC_s = rankC[ordC]
            posC = torch.searchsorted(keysC_s, keys_u).clamp(max=keysC_s.numel() - 1)
            is_center = keysC_s[posC] == keys_u
            rank_center = torch.where(is_center, rankC_s[posC], torch.full_like(posC, -1))
        else:
            is_center = torch.zeros_like(keys_u, dtype=torch.bool)
            rank_center = torch.full_like(keys_u, -1)
        del slotC, gC, rankC, keysC, keys_u
        dA = depths_A[gc]
        # cap to K: sort by (pixel, [centre-first], A-depth), keep rank < K
        idx = torch.argsort(dA)
        if args.cap_policy == "center_depth":
            idx = idx[torch.argsort((~is_center[idx]).to(torch.int8), stable=True)]
        idx = idx[torch.argsort(pc[idx], stable=True)]
        pc, gc, dA, is_center, rank_center = pc[idx], gc[idx], dA[idx], is_center[idx], rank_center[idx]
        cnt = torch.bincount(pc - p0, minlength=Pc)
        rank_in_pix = torch.arange(pc.numel(), device=device) - (torch.cumsum(cnt, 0) - cnt)[pc - p0]
        n_cap_bind += int((cnt > K_cap).sum().item())
        keep = rank_in_pix < K_cap
        pc, gc, dA, is_center, rank_center = pc[keep], gc[keep], dA[keep], is_center[keep], rank_center[keep]
        del idx, keep, rank_in_pix, cnt
        n_cand_total += int(pc.numel())
        if pc.numel() == 0:
            continue

        # fresh alpha along B's rays; fresh order by B depth
        alpha = eval_alpha(stB, gc, pc, W)
        dB = depths_B[gc]
        idx = lexsort_by_pixel(pc, dB)
        pcs, gcs, alphas = pc[idx], gc[idx], alpha[idx]
        offs_c, _ = csr_offsets(pcs - p0, Pc)
        _, w = segment_weights(alphas, offs_c)
        rgb, acc = composite(w, gcs, pcs, col_B, H, W)
        oracle_rgb += rgb
        oracle_alpha += acc
        rgb_sh, _ = composite(w, gcs, pcs, col_A, H, W)  # C1: stale SH
        stale_sh_rgb += rgb_sh
        del rgb, acc, rgb_sh, w, pcs, gcs, alphas, idx, dB

        # B4 variant: A's stale order (centre-list rank, then neighbours by A depth)
        ordval = torch.where(is_center, rank_center.to(torch.float32), 1.0e6 + dA)
        idx = lexsort_by_pixel(pc, ordval)
        pcs, gcs, alphas = pc[idx], gc[idx], alpha[idx]
        offs_c, _ = csr_offsets(pcs - p0, Pc)
        _, w = segment_weights(alphas, offs_c)
        rgb, _ = composite(w, gcs, pcs, col_B, H, W)
        stale_order_rgb += rgb
        del rgb, w, pcs, gcs, alphas, idx, ordval, alpha

        # B5: union and 2x2 quad sharing
        union_mask[gc] = True
        ci = torch.div(pc, W, rounding_mode="floor")
        cj = pc % W
        inq = (ci < Hq2) & (cj < Wq2)
        quad = torch.div(ci, 2, rounding_mode="floor") * Wq + torch.div(cj, 2, rounding_mode="floor")
        qk = (quad * N + gc)[inq]
        quad_num += int(torch.unique(qk).numel())
        quad_den += int(inq.sum().item())
        del pc, gc, dA, is_center, rank_center, ci, cj, inq, quad, qk

    # ---- metrics -------------------------------------------------------------------
    res: Dict[str, float] = {}
    valid_img = valid.reshape(H, W)
    cyc_img = valid_cycle.reshape(H, W)
    res["valid_frac"] = float(valid.float().mean().item())
    res["has_corr_frac"] = float(has_corr.float().mean().item())
    res["valid_cycle_frac"] = float(valid_cycle.float().mean().item())
    iv = iou[valid]
    if iv.numel() > 0:
        res["iou_mean"] = float(iv.mean().item())
        res["iou_p10"] = float(iv.kthvalue(max(1, int(0.1 * iv.numel()))).values.item())
        res["iou_p50"] = float(iv.median().item())
    else:
        res["iou_mean"] = res["iou_p10"] = res["iou_p50"] = NAN
    ic = iou[valid_cycle]
    if ic.numel() > 0:
        res["iou_mean_cycle"] = float(ic.mean().item())
        res["iou_p10_cycle"] = float(ic.kthvalue(max(1, int(0.1 * ic.numel()))).values.item())
        res["iou_p50_cycle"] = float(ic.median().item())
    else:
        res["iou_mean_cycle"] = res["iou_p10_cycle"] = res["iou_p50_cycle"] = NAN
    res["cap_bind_frac"] = n_cap_bind / max(n_cand_pixels, 1)
    res["mean_candidates_per_pixel"] = n_cand_total / max(n_cand_pixels, 1)
    res["n_candidate_pairs"] = n_cand_total
    res["n_shared_pairs"] = n_shared_total

    res["psnr_oracle"] = psnr(oracle_rgb, full_rgb)
    res["psnr_oracle_valid"] = psnr(oracle_rgb[valid_img], full_rgb[valid_img]) if bool(valid.any()) else NAN
    res["psnr_stale_order"] = psnr(stale_order_rgb, full_rgb)
    res["psnr_stale_sh"] = psnr(stale_sh_rgb, full_rgb)
    res["psnr_stale_order_valid"] = psnr(stale_order_rgb[valid_img], full_rgb[valid_img]) if bool(valid.any()) else NAN
    res["psnr_stale_sh_valid"] = psnr(stale_sh_rgb[valid_img], full_rgb[valid_img]) if bool(valid.any()) else NAN
    if lp is not None:
        res["lpips_oracle"] = lp(oracle_rgb, full_rgb)
        m3 = valid_img[..., None].float()
        res["lpips_oracle_valid_masked"] = lp(oracle_rgb * m3, full_rgb * m3)
    else:
        res["lpips_oracle"] = NAN
        res["lpips_oracle_valid_masked"] = NAN
    fb_masks = {}
    for t in taus:
        m = oracle_alpha < (full_alpha - t)
        fb_masks[t] = m
        res[tau_col(t)] = float(m.float().mean().item())
    res["inversion_frac"] = n_disc / max(n_pairs_total, 1)
    if gap_chunks:
        gaps = torch.cat(gap_chunks)
        res["inv_depth_gap_p50"] = float(gaps.median().item())
    else:
        res["inv_depth_gap_p50"] = NAN
    res["union_frac"] = float(union_mask.sum().item()) / max(N, 1)
    res["quad_share"] = quad_num / max(quad_den, 1)
    res["N"] = N
    res["time_s"] = time.perf_counter() - t_start
    _ = cyc_img

    # ---- PNGs -------------------------------------------------------------------------
    if png_prefix is not None:
        t_max = max(taus)
        save_png(png_prefix + "_fallback.png", fb_masks[t_max].float())
        if not png_fallback_only:
            save_png(png_prefix + "_B_full.png", full_rgb)
            save_png(png_prefix + "_oracle.png", oracle_rgb)
            save_png(png_prefix + "_err10.png", (oracle_rgb - full_rgb).abs() * 10.0)

    del stA, stB, SA, SB, iou, oracle_rgb, oracle_alpha, stale_order_rgb, stale_sh_rgb, fb_masks, gap_chunks
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_pairs(views: List[int], c2ws: np.ndarray, Ks: List[Tensor], scene_scale: float,
                kinds: List[str], selfcheck: bool) -> List[dict]:
    """List of pair specs: view, kind, delta, c2w_A, c2w_B, K_A, K_B, png flags."""
    pairs: List[dict] = []
    first = views[0]
    if selfcheck:
        # (a) exactness check: no eps truncation, no K cap -> the oracle must
        #     reproduce B's render up to float error; (b) the same identity pair
        #     at the run's eps / K, which documents the truncation floor at delta 0.
        pairs.append(dict(view=first, kind="identity_exact", delta=0.0, c2w_A=c2ws[first], c2w_B=c2ws[first].copy(),
                          K_A=Ks[0], K_B=Ks[0], png=None, eps=0.0, K=1 << 20))
        pairs.append(dict(view=first, kind="identity", delta=0.0, c2w_A=c2ws[first], c2w_B=c2ws[first].copy(),
                          K_A=Ks[0], K_B=Ks[0], png="full"))
    for vi, v in enumerate(views):
        c2w_A = c2ws[v].astype(np.float64)
        K_v = Ks[vi]
        sweeps = [
            ("rot_yaw", ROT_DELTAS_DEG, lambda d: local_delta(c2w_A, R=rot_y(d))),
            ("rot_pitch", ROT_DELTAS_DEG, lambda d: local_delta(c2w_A, R=rot_x(d))),
            ("trans_x", TRANS_DELTAS_PCT, lambda d: local_delta(c2w_A, t=np.array([d / 100.0 * scene_scale, 0.0, 0.0]))),
            ("trans_z", TRANS_DELTAS_PCT, lambda d: local_delta(c2w_A, t=np.array([0.0, 0.0, d / 100.0 * scene_scale]))),
        ]
        for kind, deltas, make in sweeps:
            if kind not in kinds:
                continue
            for di, d in enumerate(deltas):
                png = None
                if vi == 0 and di == len(deltas) // 2:
                    png = "full"
                elif vi == 0 and di == len(deltas) - 1:
                    png = "fallback"
                pairs.append(dict(view=v, kind=kind, delta=float(d), c2w_A=c2w_A, c2w_B=make(d),
                                  K_A=K_v, K_B=K_v, png=png))
    if "traj" in kinds:
        if len(views) >= 2:
            tp, n_interp = trajectory_pairs(c2ws[views], scene_scale, N_TRAJ_PAIRS)
            print(f"[traj] n_interp={n_interp}, {len(tp)} consecutive pairs")
            for k, (a, b) in enumerate(tp):
                png = "full" if k == len(tp) // 2 else ("fallback" if k == len(tp) - 1 else None)
                pairs.append(dict(view=first, kind="traj", delta=float(k + 1), c2w_A=a, c2w_B=b,
                                  K_A=Ks[0], K_B=Ks[0], png=png))
        else:
            print("[traj] skipped: needs >= 2 base views")
    return pairs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--views", type=str, default="0,40", help="comma-separated base camera indices (default 0,40; full run e.g. 0,40,80)")
    ap.add_argument("--res", type=str, default="native",
                    help="'native' (dataset resolution at data_factor), explicit WxH (aspect not preserved; "
                         "K scaled per axis), or a target width with height by aspect ratio. "
                         "Default native for speed; use 1920 for the full run")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--tau", type=str, default="0.01,0.05")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "out"))
    ap.add_argument("--kinds", type=str, default=",".join(KINDS_ALL))
    ap.add_argument("--cap_policy", choices=["center_depth", "depth"], default="center_depth")
    ap.add_argument("--strict_valid", action="store_true", help="oracle only for valid (acc_alpha_B > 0.5) pixels")
    ap.add_argument("--chunk_px", type=int, default=131072, help="approx pixels per processing chunk")
    ap.add_argument("--inv_pad", type=int, default=64, help="pad/truncate shared lists to this many for inversions")
    ap.add_argument("--inv_sub", type=int, default=8192, help="pixels per pairwise-comparison block")
    ap.add_argument("--gap_samples", type=int, default=1_000_000, help="max depth-gap samples kept per block")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="use common.make_synthetic_scene (256x256, single view)")
    ap.add_argument("--synthetic_sh", type=int, default=None, help="SH degree for the synthetic scene (default: RGB)")
    ap.add_argument("--synthetic_views", type=int, default=1, help="number of synthetic base views (>=2 enables traj)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true",
                    help="add identity pairs; the one at eps=0 / uncapped K must be exact (PASS/FAIL)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    taus = [float(t) for t in args.tau.split(",") if t.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    os.makedirs(args.out, exist_ok=True)
    png_dir = os.path.join(args.out, "png")
    os.makedirs(png_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- scene ------------------------------------------------------------------------
    if args.synthetic:
        splats, viewmat, K0, W, H = make_synthetic_scene(args.seed, sh_degree=args.synthetic_sh)
        sh_degree = args.synthetic_sh
        scene_scale = 1.0
        c2w0 = np.linalg.inv(viewmat.double().cpu().numpy())
        # optional extra base views (small yaw + x-translation steps) so the
        # trajectory code path can be exercised without a dataset
        c2ws = np.stack(
            [local_delta(c2w0, R=rot_y(3.0 * k), t=np.array([0.05 * k, 0.0, 0.0])) for k in range(max(1, args.synthetic_views))]
        )
        views = list(range(len(c2ws)))
        Ks = [K0.to(device) for _ in views]
        print(f"[scene] synthetic seed={args.seed} N={splats['N']} {W}x{H} sh_degree={sh_degree} scene_scale={scene_scale}")
    else:
        assert args.ckpt and args.data_dir, "--ckpt and --data_dir required (or --synthetic)"
        splats = load_splats(args.ckpt, device)
        sh_degree = splats["sh_degree"]
        cams = load_cameras(args.data_dir, args.data_factor)
        c2ws = cams["camtoworlds"].astype(np.float64)
        scene_scale = cams["scene_scale"]
        views = [int(v) for v in args.views.split(",") if v.strip()]
        for v in views:
            assert 0 <= v < len(c2ws), f"view {v} out of range (n={len(c2ws)})"
        w0, h0 = cams["imsize_dict"][cams["camera_ids"][views[0]]]
        if str(args.res).lower() == "native":
            W, H = int(w0), int(h0)
        elif "x" in str(args.res).lower():
            W, H = (int(t) for t in str(args.res).lower().split("x"))
        else:
            W = int(args.res)
            H = int(round(h0 * W / w0 / 2.0) * 2)
        Ks = []
        for v in views:
            cid = cams["camera_ids"][v]
            wv, hv = cams["imsize_dict"][cid]
            Kv = scale_K(cams["Ks_dict"][cid], wv, hv, W, H)
            Ks.append(torch.as_tensor(Kv, dtype=torch.float32, device=device))
        print(f"[scene] ckpt={args.ckpt} N={splats['N']} sh_degree={sh_degree} res={W}x{H} "
              f"(native {w0}x{h0}) scene_scale={scene_scale:.4f} views={views}")

    lp = None
    if not args.no_lpips:
        try:
            lp = lpips_fn(device)
        except Exception as e:  # pragma: no cover - offline weights
            print(f"[warn] LPIPS unavailable ({e}); lpips columns will be nan")

    pairs = build_pairs(views, c2ws, Ks, scene_scale, kinds, args.selfcheck)
    print(f"[pairs] {len(pairs)} pose pairs; eps={args.eps} K={args.K} radius={args.radius} taus={taus} "
          f"cap_policy={args.cap_policy} strict_valid={args.strict_valid}")

    columns = CSV_COLUMNS_FIXED + [tau_col(t) for t in taus] + CSV_COLUMNS_TAIL
    csv_path = os.path.join(args.out, "temporal_pairs.csv")
    extra_path = os.path.join(args.out, "temporal_pairs_extra.csv")
    f_csv = open(csv_path, "w", newline="")
    f_ext = open(extra_path, "w", newline="")
    w_csv = csv.writer(f_csv)
    w_ext = csv.writer(f_ext)
    w_csv.writerow(columns)
    w_ext.writerow(EXTRA_COLUMNS)

    selfcheck_rows: List[Dict[str, float]] = []
    for k, pr in enumerate(pairs):
        rot_deg, trans_frac = pose_delta(pr["c2w_A"], pr["c2w_B"], scene_scale)
        png_prefix = None
        fb_only = False
        if pr["png"] is not None:
            png_prefix = os.path.join(png_dir, f"v{pr['view']}_{pr['kind']}_{pr['delta']:g}")
            fb_only = pr["png"] == "fallback"
        eps_k = float(pr.get("eps", args.eps))
        K_k = int(pr.get("K", args.K))
        res = measure_pair(
            splats, sh_degree, pr["c2w_A"], pr["c2w_B"], pr["K_A"], pr["K_B"], W, H,
            args, lp, taus, eps_k, K_k, png_prefix=png_prefix, png_fallback_only=fb_only,
        )
        row = {
            "view": pr["view"], "kind": pr["kind"], "delta": pr["delta"],
            "rot_deg": rot_deg, "trans_frac": trans_frac, "N": res["N"],
            "eps": eps_k, "K": K_k, "radius": args.radius,
        }
        row.update({c: res[c] for c in columns if c not in row})
        w_csv.writerow([fmt(row[c]) for c in columns])
        f_csv.flush()
        ext = {"view": pr["view"], "kind": pr["kind"], "delta": pr["delta"], "cap_policy": args.cap_policy}
        ext.update({c: res[c] for c in EXTRA_COLUMNS if c not in ext})
        w_ext.writerow([fmt(ext[c]) for c in EXTRA_COLUMNS])
        f_ext.flush()
        fb_str = " ".join(f"fb{t:g}={res[tau_col(t)]:.4f}" for t in taus)
        print(
            f"[{k + 1:3d}/{len(pairs)}] v{pr['view']} {pr['kind']:<9s} d={pr['delta']:<5g} "
            f"rot={rot_deg:.3f}deg trans={trans_frac:.4f} | valid={res['valid_frac']:.3f} "
            f"iou p50={res['iou_p50']:.4f} mean={res['iou_mean']:.4f} (cycle p50={res['iou_p50_cycle']:.4f}) "
            f"| psnr={res['psnr_oracle']:.2f} valid={res['psnr_oracle_valid']:.2f} lpips={res['lpips_oracle']:.4f} "
            f"| {fb_str} | inv={res['inversion_frac']:.4f} gap50={res['inv_depth_gap_p50']:.4f} "
            f"stale_order={res['psnr_stale_order']:.2f} stale_sh={res['psnr_stale_sh']:.2f} "
            f"| cand/px={res['mean_candidates_per_pixel']:.1f} cap={res['cap_bind_frac']:.3f} "
            f"union={res['union_frac']:.3f} quad={res['quad_share']:.3f} | {res['time_s']:.1f}s"
        )
        if pr["kind"] == "identity_exact":
            selfcheck_rows.append(res)
        elif pr["kind"] == "identity":
            print(
                f"[selfcheck] info: identity pair at eps={eps_k:g} K={K_k} (truncation floor at delta 0): "
                f"psnr_oracle={res['psnr_oracle']:.2f} dB, "
                + ", ".join(f"{tau_col(t)}={res[tau_col(t)]:.4f}" for t in taus)
                + f", cap_bind_frac={res['cap_bind_frac']:.3f}"
            )
    f_csv.close()
    f_ext.close()
    print(f"[out] wrote {csv_path}")
    print(f"[out] wrote {extra_path}")
    print(f"[out] PNGs in {png_dir}")

    if args.selfcheck:
        ok_all = True
        for res in selfcheck_rows:
            checks = [
                ("iou_p50 == 1.0", res["iou_p50"] == 1.0, res["iou_p50"]),
                ("psnr_oracle >= 50 dB", res["psnr_oracle"] >= 50.0, res["psnr_oracle"]),
                ("psnr_oracle_valid >= 50 dB", res["psnr_oracle_valid"] >= 50.0, res["psnr_oracle_valid"]),
                ("inversion_frac == 0", res["inversion_frac"] == 0.0, res["inversion_frac"]),
            ]
            for t in taus:
                checks.append((f"{tau_col(t)} == 0", res[tau_col(t)] == 0.0, res[tau_col(t)]))
            for name, ok, val in checks:
                ok_all &= bool(ok)
                print(f"[selfcheck] {'PASS' if ok else 'FAIL'}  {name:<32s} value={fmt(float(val))}")
        print(f"[selfcheck] {'PASS' if ok_all else 'FAIL'} (identity pair, eps=0, uncapped K)")
        if not ok_all:
            sys.exit(1)


if __name__ == "__main__":
    main()
