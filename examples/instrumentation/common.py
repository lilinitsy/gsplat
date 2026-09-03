"""
Instrumentation code written by Fable 5.1.
Conventions
* Single camera, non-packed, no batch dims: per-view tensors are [1, N, ...].
* ``tile_size = 16`` unless stated otherwise.
* All pixel-space quantities use the kernel convention (px = j + 0.5, py = i + 0.5) with ``i`` the row (y) and ``j`` the column (x).
* ``ray_transforms[g]`` is used *row-wise* exactly as the CUDA kernels do (``u_M = M[0]``, ``v_M = M[1]``, ``w_M = M[2]``; ``h_u = px * w_M - u_M``).
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.cuda._wrapper import (
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)

# Constants mirrored from gsplat/cuda/include/Common.h and
# gsplat/cuda/csrc/Rasterization.h
ALPHA_THRESHOLD: float = 1.0 / 255.0
FILTER_INV_SQUARE_2DGS: float = 2.0
ALPHA_MAX: float = 0.999
T_DONE: float = 1e-4 

_EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@torch.no_grad()
def load_splats(ckpt_path: str, device) -> dict:
    """Load a ``simple_trainer_2dgs.py`` checkpoint and activate parameters.

    Returns dict with keys ``means [N,3]``, ``quats [N,4]`` (normalized),
    ``scales [N,3]`` (exp), ``opacities [N]`` (sigmoid), ``colors [N,K,3]`` (SH
    coefficients), ``sh_degree`` (int) and ``N``.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:  # older pickles / non-tensor payloads
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sp = ckpt["splats"]
    means = sp["means"].to(device).float()
    quats = F.normalize(sp["quats"].to(device).float(), dim=-1)
    scales = torch.exp(sp["scales"].to(device).float())
    opacities = torch.sigmoid(sp["opacities"].to(device).float())
    colors = torch.cat([sp["sh0"], sp["shN"]], dim=-2).to(device).float()
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
        "N": int(means.shape[0]),
    }


def load_cameras(data_dir: str, data_factor: int) -> dict:
    """Load COLMAP cameras exactly as ``simple_trainer_2dgs.py`` does.

    Parser is constructed with ``factor=data_factor, normalize=True,
    test_every=8`` (the trainer's ``Config`` defaults), so world coordinates
    match a checkpoint trained with default settings.

    Returns dict with ``camtoworlds [n,4,4]`` float32 np, ``K [3,3]`` np (first
    camera), ``width``, ``height`` (first camera), ``scene_scale``, plus
    ``camera_ids``, ``Ks_dict``, ``imsize_dict``, ``image_names``.
    """
    if _EXAMPLES_DIR not in sys.path:
        sys.path.insert(0, _EXAMPLES_DIR)
    from datasets.colmap import Parser  # noqa: WPS433 (examples/ import)

    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=True,
        test_every=8,
    )
    cam0 = parser.camera_ids[0]
    K = np.asarray(parser.Ks_dict[cam0], dtype=np.float32).copy()
    width, height = parser.imsize_dict[cam0]
    return {
        "camtoworlds": np.asarray(parser.camtoworlds, dtype=np.float32),
        "K": K,
        "width": int(width),
        "height": int(height),
        "scene_scale": float(parser.scene_scale),
        "camera_ids": list(parser.camera_ids),
        "Ks_dict": {k: np.asarray(v, dtype=np.float32) for k, v in parser.Ks_dict.items()},
        "imsize_dict": dict(parser.imsize_dict),
        "image_names": list(parser.image_names),
    }


def scale_K(K, w: int, h: int, target_w: int, target_h: int):
    """Rescale intrinsics from a ``(w, h)`` image to ``(target_w, target_h)``."""
    is_t = torch.is_tensor(K)
    K2 = K.clone() if is_t else np.array(K, copy=True)
    sx = target_w / float(w)
    sy = target_h / float(h)
    K2[0, 0] = K[0, 0] * sx
    K2[0, 2] = K[0, 2] * sx
    K2[1, 1] = K[1, 1] * sy
    K2[1, 2] = K[1, 2] * sy
    return K2


def crop_K(K, x0: int, y0: int):
    """Intrinsics for the sub-image whose top-left pixel is ``(x0, y0)``."""
    is_t = torch.is_tensor(K)
    K2 = K.clone() if is_t else np.array(K, copy=True)
    K2[0, 2] = K[0, 2] - x0
    K2[1, 2] = K[1, 2] - y0
    return K2


# --------------------------------------------------------------------------- #
# Rendering by stage
# --------------------------------------------------------------------------- #
def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _timed(fn: Callable[[], object], n_warmup: int, n_iter: int) -> Tuple[object, float]:
    """Run ``fn`` n_warmup + n_iter times; return (last result, median ms)."""
    for _ in range(n_warmup):
        fn()
    ts = []
    out = None
    for _ in range(n_iter):
        t0 = _sync_time()
        out = fn()
        t1 = _sync_time()
        ts.append((t1 - t0) * 1e3)
    return out, float(np.median(ts))


@torch.no_grad()
def render_stages(
    splats: dict,
    viewmat: Tensor,
    K: Tensor,
    W: int,
    H: int,
    *,
    sh_degree: Optional[int] = None,
    timing: bool = False,
    n_warmup: int = 2,
    n_iter: int = 5,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
) -> dict:
    """Re-implementation of ``gsplat.rendering.rasterization_2dgs`` (non-packed,
    C=1, no batch dims, render_mode="RGB+D") with every stage exposed.

    ``splats["colors"]`` is ``[N,3]`` RGB when ``sh_degree is None`` else
    ``[N,K,3]`` SH coefficients.

    Returns a dict with: radii[1,N,2], means2d[1,N,2], depths[1,N],
    ray_transforms[1,N,3,3], normals[1,N,3], opacities[1,N], colors_rgb[1,N,3]
    (post-SH, ``clamp_min(sh+0.5, 0)``), tile_width, tile_height,
    tiles_per_gauss, isect_ids, flatten_ids, isect_offsets[1,th,tw],
    render_colors[1,H,W,3], render_alphas[1,H,W,1], render_depth[1,H,W,1]
    (accumulated depth, un-normalized), render_median[1,H,W,1] (median depth),
    render_normals[1,H,W,3] (camera space), W, H, tile_size, N, n_isects, and
    if timing: times_ms = {projection, tiling_sort, sh, blend, total}.
    """
    device = splats["means"].device
    means = splats["means"]
    quats = splats["quats"]
    scales = splats["scales"]
    opacities = splats["opacities"]
    colors_in = splats["colors"]
    N = means.shape[0]

    viewmats = torch.as_tensor(viewmat, dtype=torch.float32, device=device).reshape(1, 4, 4)
    Ks = torch.as_tensor(K, dtype=torch.float32, device=device).reshape(1, 3, 3)
    if sh_degree is not None:
        assert colors_in.dim() == 3 and colors_in.shape[-1] == 3, colors_in.shape
        assert (sh_degree + 1) ** 2 <= colors_in.shape[-2], colors_in.shape
    else:
        assert colors_in.dim() == 2 and colors_in.shape == (N, 3), colors_in.shape

    tile_width = math.ceil(W / float(tile_size))
    tile_height = math.ceil(H / float(tile_size))

    # ---- stage 1: projection ------------------------------------------------
    def st_projection():
        return fully_fused_projection_2dgs(
            means, quats, scales, viewmats, Ks, W, H,
            eps2d, near_plane, far_plane, radius_clip, False, False,
        )

    # ---- stage 2: tiling + sort --------------------------------------------
    def st_tiling(proj):
        radii, means2d, depths, _, _ = proj
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            packed=False, n_images=1, image_ids=None, gaussian_ids=None,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
        isect_offsets = isect_offsets.reshape(1, tile_height, tile_width)
        return tiles_per_gauss, isect_ids, flatten_ids, isect_offsets

    # ---- stage 3: SH ----------------------------------------------------------
    def st_sh(proj):
        radii = proj[0]
        if sh_degree is None:
            return colors_in[None].contiguous()  # [1,N,3]
        camtoworlds = torch.inverse(viewmats)
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [1,N,3]
        shs = torch.broadcast_to(colors_in[None], (1, N) + tuple(colors_in.shape[-2:]))
        c = spherical_harmonics(sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1))
        return torch.clamp_min(c + 0.5, 0.0)

    # ---- stage 4: blend ---------------------------------------------------------
    def st_blend(proj, tiling, colors_rgb):
        radii, means2d, depths, ray_transforms, normals = proj
        _, _, flatten_ids, isect_offsets = tiling
        opac = torch.broadcast_to(opacities[None, :], (1, N))
        cols = torch.cat((colors_rgb, depths[..., None]), dim=-1)  # RGB+D
        densify = torch.zeros_like(means2d)
        return rasterize_to_pixels_2dgs(
            means2d, ray_transforms, cols, opac, normals, densify,
            W, H, tile_size, isect_offsets, flatten_ids,
            backgrounds=None, packed=False, absgrad=False, distloss=False,
        )

    times: Dict[str, float] = {}
    if timing:
        proj, times["projection"] = _timed(st_projection, n_warmup, n_iter)
        tiling, times["tiling_sort"] = _timed(lambda: st_tiling(proj), n_warmup, n_iter)
        colors_rgb, times["sh"] = _timed(lambda: st_sh(proj), n_warmup, n_iter)
        rast, times["blend"] = _timed(
            lambda: st_blend(proj, tiling, colors_rgb), n_warmup, n_iter
        )

        def st_all():
            p = st_projection()
            t = st_tiling(p)
            c = st_sh(p)
            return st_blend(p, t, c)

        _, times["total"] = _timed(st_all, n_warmup, n_iter)
    else:
        proj = st_projection()
        tiling = st_tiling(proj)
        colors_rgb = st_sh(proj)
        rast = st_blend(proj, tiling, colors_rgb)

    radii, means2d, depths, ray_transforms, normals = proj
    tiles_per_gauss, isect_ids, flatten_ids, isect_offsets = tiling
    render_colors, render_alphas, render_normals, render_distort, render_median = rast

    # gsplat allocates the per-view projection/SH outputs with at::empty and its
    # kernels return early for culled gaussians (radii == 0), so those entries hold
    # uninitialized memory. The full-frame render never reads them, but anything
    # evaluating arbitrary (pixel, gaussian) pairs — the oracle warp — does. Zero
    # them (a zero ray transform gives cross.z == 0 -> alpha 0 in eval_alpha).
    vis = (radii > 0).all(dim=-1)  # [1,N]
    means2d = torch.where(vis[..., None], means2d, torch.zeros_like(means2d))
    depths = torch.where(vis, depths, torch.zeros_like(depths))
    ray_transforms = torch.where(
        vis[..., None, None], ray_transforms, torch.zeros_like(ray_transforms)
    )
    colors_rgb = torch.where(vis[..., None], colors_rgb, torch.zeros_like(colors_rgb))

    out = {
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "normals": normals,
        "opacities": torch.broadcast_to(opacities[None, :], (1, N)).contiguous(),
        "colors_rgb": colors_rgb,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_size": tile_size,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "render_colors": render_colors[..., :3].contiguous(),
        "render_depth": render_colors[..., 3:4].contiguous(),
        "render_alphas": render_alphas,
        "render_normals": render_normals,
        "render_distort": render_distort,
        "render_median": render_median,
        "W": W,
        "H": H,
        "N": N,
        "n_isects": int(flatten_ids.numel()),
        "viewmat": viewmats[0],
        "K": Ks[0],
    }
    if timing:
        out["times_ms"] = times
    return out


@torch.no_grad()
def run_full_render(
    splats: dict, viewmat: Tensor, K: Tensor, W: int, H: int, *, tile_size: int = 16
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Reference render through ``gsplat.rendering.rasterization_2dgs``.

    Returns ``(render_colors[1,H,W,3], render_alphas[1,H,W,1],
    render_median[1,H,W,1], meta)``.  Used by tests to cross-check
    :func:`render_stages`.
    """
    from gsplat.rendering import rasterization_2dgs

    device = splats["means"].device
    viewmats = torch.as_tensor(viewmat, dtype=torch.float32, device=device).reshape(1, 4, 4)
    Ks = torch.as_tensor(K, dtype=torch.float32, device=device).reshape(1, 3, 3)
    sh_degree = splats.get("sh_degree", None)
    colors = splats["colors"]
    if sh_degree is None:
        colors = colors[None]  # [C=1, N, 3]; a bare [N,3] fails the wrapper's assert
    rc, ra, _, _, _, rmed, meta = rasterization_2dgs(
        splats["means"], splats["quats"], splats["scales"], splats["opacities"],
        colors, viewmats, Ks, W, H,
        sh_degree=sh_degree, packed=False, tile_size=tile_size, render_mode="RGB+D",
    )
    return rc[..., :3].contiguous(), ra, rmed, meta


# --------------------------------------------------------------------------- #
# Contributor extraction and re-compositing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_contributors(stage: dict, W: int, H: int) -> dict:
    """Per-pixel front-to-back contributor lists from the CUDA index kernel.

    Calls ``rasterize_to_indices_in_range_2dgs(0, 1<<30, T=ones, ...)``, which
    returns, for every pixel, the surfels with ``alpha >= 1/255`` in depth order,
    stopping before the surfel that would drive ``T <= 1e-4``.

    Returns dict: ``gs_ids [M] int64``, ``pix_ids [M] int64`` (row-major,
    non-decreasing), ``offsets [H*W+1] int64`` (CSR), ``n_contrib [H*W] int64``,
    ``sorted_by_us`` (bool; True only if the kernel output had to be re-sorted).
    """
    device = stage["means2d"].device
    trans = torch.ones((1, H, W), dtype=torch.float32, device=device)
    gs_ids, pix_ids, img_ids = rasterize_to_indices_in_range_2dgs(
        0, 1 << 30, trans,
        stage["means2d"], stage["ray_transforms"], stage["opacities"],
        W, H, stage["tile_size"], stage["isect_offsets"], stage["flatten_ids"],
    )
    gs_ids = gs_ids.to(torch.int64)
    pix_ids = pix_ids.to(torch.int64)
    sorted_by_us = False
    if pix_ids.numel() > 1:
        nondecreasing = bool((pix_ids[1:] >= pix_ids[:-1]).all().item())
        if not nondecreasing:
            # The kernel writes at chunk_starts[pix] + cnt, so this should never
            # trigger; fall back to a stable sort (preserves depth order within a pixel).
            order = torch.sort(pix_ids, stable=True).indices
            gs_ids, pix_ids = gs_ids[order], pix_ids[order]
            sorted_by_us = True
    n_contrib = torch.bincount(pix_ids, minlength=H * W)
    offsets = torch.zeros(H * W + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(n_contrib, dim=0)
    assert int(offsets[-1].item()) == gs_ids.numel()
    return {
        "gs_ids": gs_ids,
        "pix_ids": pix_ids,
        "offsets": offsets,
        "n_contrib": n_contrib,
        "sorted_by_us": sorted_by_us,
    }


@torch.no_grad()
def eval_alpha(
    stage: dict,
    gs_ids: Tensor,
    pix_ids: Tensor,
    W: int,
    *,
    apply_threshold: bool = True,
    chunk: int = 8_000_000,
) -> Tensor:
    """Kernel-exact alpha for each ``(pixel, gaussian)`` pair.

    ``alpha = min(0.999, opac * exp(-0.5 * min(u^2+v^2, 2*|means2d - p|^2)))``
    with ``(u, v)`` the ray-splat intersection from ``ray_transforms``.
    Pairs the kernel would skip (``cross.z == 0``, ``sigma < 0``/NaN, and — if
    ``apply_threshold`` — ``alpha < 1/255``) get ``alpha = 0``.
    Works for arbitrary pairs, not just those returned by the index kernel.
    """
    means2d = stage["means2d"][0]  # [N,2]
    M_all = stage["ray_transforms"][0]  # [N,3,3]
    opac_all = stage["opacities"][0]  # [N]
    # Culled gaussians (radii == 0) have uninitialized per-view state in gsplat's
    # buffers (at::empty, kernel returns early). render_stages zeroes that state,
    # which alone yields alpha 0 here; the radii mask is a second guard for stage
    # dicts that still carry it (callers may prune it to save memory).
    vis_all = (stage["radii"][0] > 0).all(dim=-1) if "radii" in stage else None  # [N]
    out = torch.empty(gs_ids.shape[0], dtype=torch.float32, device=gs_ids.device)
    for s in range(0, gs_ids.shape[0], chunk):
        g = gs_ids[s : s + chunk]
        p = pix_ids[s : s + chunk]
        px = (p % W).to(torch.float32) + 0.5
        py = torch.div(p, W, rounding_mode="floor").to(torch.float32) + 0.5
        Mg = M_all[g]  # [m,3,3]
        u_M, v_M, w_M = Mg[:, 0, :], Mg[:, 1, :], Mg[:, 2, :]
        h_u = px[:, None] * w_M - u_M
        h_v = py[:, None] * w_M - v_M
        c = torch.cross(h_u, h_v, dim=-1)
        cz = c[:, 2]
        safe = cz != 0
        cz_safe = torch.where(safe, cz, torch.ones_like(cz))
        uu = c[:, 0] / cz_safe
        vv = c[:, 1] / cz_safe
        g3d = uu * uu + vv * vv
        d = means2d[g] - torch.stack([px, py], dim=-1)
        g2d = FILTER_INV_SQUARE_2DGS * (d[:, 0] ** 2 + d[:, 1] ** 2)
        sigma = 0.5 * torch.minimum(g3d, g2d)
        alpha = torch.clamp_max(opac_all[g] * torch.exp(-sigma), ALPHA_MAX)
        valid = safe & (sigma >= 0) & torch.isfinite(alpha)
        if vis_all is not None:
            valid = valid & vis_all[g]
        if apply_threshold:
            valid = valid & (alpha >= ALPHA_THRESHOLD)
        out[s : s + chunk] = torch.where(valid, alpha, torch.zeros_like(alpha))
    return out


def _segment_ids_from_offsets(offsets: Tensor) -> Tensor:
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=offsets.device, dtype=torch.int64), counts
    )


@torch.no_grad()
def segment_weights(alpha: Tensor, offsets: Tensor) -> Tuple[Tensor, Tensor]:
    """Front-to-back transmittance and weights within each CSR segment.

    ``T_i = prod_{j<i, same segment} (1 - alpha_j)``, ``w_i = alpha_i * T_i``.
    Implemented as a float64 log-space cumsum minus the value at segment start.
    """
    M = alpha.shape[0]
    if M == 0:
        z = alpha.new_zeros(0)
        return z, z
    a = alpha.to(torch.float64).clamp(max=ALPHA_MAX)
    log1m = torch.log1p(-a)
    c_incl = torch.cumsum(log1m, dim=0)
    c_excl = c_incl - log1m
    seg = _segment_ids_from_offsets(offsets)  # [M]
    start = c_excl[offsets[:-1].clamp(max=M - 1)][seg]  # value at each segment's first element
    logT = c_excl - start
    T = torch.exp(logT).to(torch.float32)
    w = (alpha * T).to(torch.float32)
    return T, w


@torch.no_grad()
def composite(
    w: Tensor, gs_ids: Tensor, pix_ids: Tensor, colors_rgb: Tensor, H: int, W: int
) -> Tuple[Tensor, Tensor]:
    """Scatter-add ``w * color`` per pixel. No background. Returns
    ``(rgb [H,W,3], acc_alpha [H,W])``."""
    if colors_rgb.dim() == 3:
        colors_rgb = colors_rgb[0]
    C = colors_rgb.shape[-1]
    rgb = torch.zeros((H * W, C), dtype=torch.float32, device=w.device)
    acc = torch.zeros((H * W,), dtype=torch.float32, device=w.device)
    rgb.index_add_(0, pix_ids, w[:, None] * colors_rgb[gs_ids].float())
    acc.index_add_(0, pix_ids, w)
    return rgb.reshape(H, W, C), acc.reshape(H, W)


# --------------------------------------------------------------------------- #
# Evaluated vs. contributing counts (A2)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluated_counts(
    stage: dict,
    gs_ids: Tensor,
    pix_ids: Tensor,
    offsets: Tensor,
    W: int,
    H: int,
    tile_size: int,
    *,
    scan_chunk: int = 32,
    pix_chunk: int = 262_144,
    max_steps: int = 100_000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """How many tile-list entries the CUDA rasterizer *evaluated* per pixel.

    The kernel walks its tile's list ``flatten_ids[off[t]:off[t+1]]`` front to
    back and stops *before* the first surfel (with ``alpha >= 1/255``) that would
    make ``T <= 1e-4``.  So ``n_eval`` is the 1-based position of that
    terminating surfel if the pixel saturated, else ``tile_len``.

    Method (vectorized): locate the pixel's last contributor in the tile list via
    keys ``tile*N + g`` + searchsorted, then scan forward over the entries after
    it (in chunks of ``scan_chunk``) evaluating kernel-exact alpha, until an entry
    with ``alpha >= 1/255`` is found (that is the terminating surfel, so the pixel
    is saturated) or the tile list ends (not saturated).  Pixels with zero
    contributors scan from the tile start.  This is exact up to float
    differences vs. the kernel; if such a difference makes a scanned entry a
    would-be contributor, it is absorbed (T updated) and the scan continues.

    Returns ``(n_eval [H*W] int64, tile_len [H*W] int64, saturated [H*W] bool)``.
    """
    device = gs_ids.device
    N = int(stage["N"])
    flatten_ids = stage["flatten_ids"].to(torch.int64)
    n_isects = int(flatten_ids.numel())
    isect_offsets = stage["isect_offsets"].reshape(-1).to(torch.int64)
    tile_width = int(stage["tile_width"])
    off_ext = torch.cat([isect_offsets, torch.tensor([n_isects], device=device, dtype=torch.int64)])
    tile_lens_t = off_ext[1:] - off_ext[:-1]  # [n_tiles]

    P = H * W
    pix = torch.arange(P, device=device, dtype=torch.int64)
    ii = torch.div(pix, W, rounding_mode="floor")
    jj = pix % W
    tile_of_pix = torch.div(ii, tile_size, rounding_mode="floor") * tile_width + torch.div(
        jj, tile_size, rounding_mode="floor"
    )
    tile_start = off_ext[tile_of_pix]  # [P]
    tile_end = off_ext[tile_of_pix + 1]  # [P]
    tile_len = tile_end - tile_start  # [P]

    counts = offsets[1:] - offsets[:-1]
    has = counts > 0

    # --- position of last contributor within its tile list ------------------
    # tile id of each isect entry
    entry_tile = torch.searchsorted(off_ext, torch.arange(n_isects, device=device), right=True) - 1
    keys = entry_tile * N + flatten_ids  # unique per (tile, g)
    keys_sorted, order = torch.sort(keys)
    last_g = gs_ids[(offsets[1:] - 1).clamp(min=0)]  # valid where has
    q = tile_of_pix * N + last_g
    pos_sorted = torch.searchsorted(keys_sorted, q).clamp(max=max(n_isects - 1, 0))
    if n_isects > 0:
        found = keys_sorted[pos_sorted] == q
        entry_idx = order[pos_sorted]  # global isect index of last contributor
    else:
        found = torch.zeros(P, dtype=torch.bool, device=device)
        entry_idx = torch.zeros(P, dtype=torch.int64, device=device)
    assert bool((found | ~has).all().item()), "last contributor not found in its tile list"

    # --- forward scan for the terminating surfel -------------------------------
    cursor = torch.where(has, entry_idx + 1, tile_start)  # next entry to examine
    T = 1.0 - stage["render_alphas"].reshape(-1).to(torch.float32)  # kernel's final T
    saturated = torch.zeros(P, dtype=torch.bool, device=device)
    n_eval = torch.zeros(P, dtype=torch.int64, device=device)
    active = cursor < tile_end
    n_eval[~active] = tile_len[~active]  # nothing left to scan: not saturated

    k = torch.arange(scan_chunk, device=device, dtype=torch.int64)

    def scan_slice(ap: Tensor) -> None:
        """One chunk-step for active pixels ``ap`` (updates state in place)."""
        cur = cursor[ap]
        end = tile_end[ap]
        idx = cur[:, None] + k[None, :]  # [A, K]
        inb = idx < end[:, None]
        g = flatten_ids[idx.clamp(max=max(n_isects - 1, 0))]  # [A, K]
        a = eval_alpha(
            stage, g.reshape(-1), ap[:, None].expand(-1, scan_chunk).reshape(-1), W,
            apply_threshold=True,
        ).reshape(-1, scan_chunk)
        hit = inb & (a > 0)  # alpha >= 1/255 (eval_alpha zeroes below-threshold)
        any_hit = hit.any(dim=1)
        first = torch.argmax(hit.to(torch.int8), dim=1)  # first hit column (valid where any_hit)
        a_first = a.gather(1, first[:, None])[:, 0]
        next_T = T[ap] * (1.0 - a_first)
        terminates = any_hit & (next_T <= T_DONE)
        absorb = any_hit & ~terminates  # float-mismatch: would have contributed
        # terminated: saturated, n_eval = 1-based position of terminating surfel
        term_pix = ap[terminates]
        saturated[term_pix] = True
        n_eval[term_pix] = (cur[terminates] + first[terminates]) - tile_start[term_pix] + 1
        active[term_pix] = False
        # absorbed: advance past it and update T
        abs_pix = ap[absorb]
        T[abs_pix] = next_T[absorb]
        cursor[abs_pix] = cur[absorb] + first[absorb] + 1
        # no hit: advance by chunk
        nh_pix = ap[~any_hit]
        cursor[nh_pix] = cur[~any_hit] + scan_chunk

    steps = 0
    while bool(active.any().item()) and steps < max_steps:
        steps += 1
        ap_all = pix[active]
        for s in range(0, ap_all.numel(), pix_chunk):
            scan_slice(ap_all[s : s + pix_chunk])
        finished = active & (cursor >= tile_end)  # ran off the tile list: not saturated
        n_eval[finished] = tile_len[finished]
        active = active & (cursor < tile_end)

    if steps >= max_steps and bool(active.any().item()):
        # give up: fall back to PLAN's heuristic for the remainder
        rem = active
        heur = T[rem] <= 2e-4
        saturated[rem] = heur
        pos = (cursor[rem] - tile_start[rem]).clamp(max=tile_len[rem] - 1) + 1
        n_eval[rem] = torch.where(heur, torch.minimum(pos + 1, tile_len[rem]), tile_len[rem])

    return n_eval, tile_len, saturated


# --------------------------------------------------------------------------- #
# Metrics / IO
# --------------------------------------------------------------------------- #
def psnr(a, b) -> float:
    """PSNR in dB between two images in [0,1] (tensor or ndarray, any shape)."""
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32).to(a_t.device)
    mse = torch.mean((a_t - b_t) ** 2).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def lpips_fn(device="cuda") -> Callable[[Tensor, Tensor], float]:
    """Returns ``f(img_a[H,W,3] in [0,1], img_b[H,W,3]) -> float`` (AlexNet LPIPS)."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    @torch.no_grad()
    def f(a: Tensor, b: Tensor) -> float:
        a_t = torch.as_tensor(a, dtype=torch.float32, device=device).clamp(0, 1)
        b_t = torch.as_tensor(b, dtype=torch.float32, device=device).clamp(0, 1)
        if a_t.dim() == 3:
            a_t = a_t[None]
            b_t = b_t[None]
        a_t = a_t.permute(0, 3, 1, 2).contiguous()
        b_t = b_t.permute(0, 3, 1, 2).contiguous()
        metric.reset()
        return float(metric(a_t, b_t).item())

    return f


def save_png(path: str, array01) -> None:
    """Save an image (values in [0,1], HxW or HxWx3, tensor or ndarray) as PNG."""
    import imageio

    arr = array01.detach().cpu().numpy() if torch.is_tensor(array01) else np.asarray(array01)
    arr = np.clip(np.nan_to_num(arr.astype(np.float32)), 0.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.imwrite(path, (arr * 255.0 + 0.5).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Synthetic scene
# --------------------------------------------------------------------------- #
def make_synthetic_scene(
    seed: int = 0,
    *,
    n: int = 20_000,
    sh_degree: Optional[int] = None,
    W: int = 256,
    H: int = 256,
    device="cuda",
) -> Tuple[dict, Tensor, Tensor, int, int]:
    """Random surfels in a 2x2x2 box 2-4 units in front of a camera at the origin.

    Returns ``(splats, viewmat[4,4], K[3,3], W, H)``. ``splats["colors"]`` is
    ``[N,3]`` RGB when ``sh_degree is None`` else random SH ``[N,(d+1)^2,3]``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.rand((n, 3), generator=gen) * 2.0 - 1.0
    means[:, 2] = means[:, 2] + 3.0  # z in [2,4]
    quats = F.normalize(torch.randn((n, 4), generator=gen), dim=-1)
    # third scale is unused by the 2DGS ray transform; keep it in-range anyway
    scales = torch.rand((n, 3), generator=gen) * 0.04 + 0.01
    opacities = torch.rand((n,), generator=gen) * 0.6 + 0.3
    if sh_degree is None:
        colors = torch.rand((n, 3), generator=gen)
    else:
        Kc = (sh_degree + 1) ** 2
        colors = torch.randn((n, Kc, 3), generator=gen) * 0.05
        colors[:, 0, :] = (torch.rand((n, 3), generator=gen) - 0.5) / 0.28209479
    splats = {
        "means": means.to(device),
        "quats": quats.to(device),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "colors": colors.to(device).contiguous(),
        "sh_degree": sh_degree,
        "N": n,
    }
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    K = torch.tensor(
        [[200.0, 0.0, W / 2.0], [0.0, 200.0, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )
    return splats, viewmat, K, W, H
