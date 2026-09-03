"""Phase 0 / WP-B: premise measurements A1 (per-stage timing) and A2
(evaluated vs. contributing surfels per pixel) for 2DGS.

Run from ``examples/`` with the gsplat env python, e.g.::

    python instrumentation/measure_premise.py --ckpt results/garden_2dgs/ckpts/ckpt_29999.pt \
        --data_dir data/360_v2/garden --data_factor 4 \
        --views 0,20,40,60,80 --resolutions native,1920,3840 --eps 0.001,0.005,0.01,0.05

    python instrumentation/measure_premise.py --synthetic --out instrumentation/out_synth   # smoke test

Outputs (in ``--out``):
    premise_timing.csv   one row per view x resolution (A1, full frame)
    premise_pixels.csv   one row per view x resolution x eps (A2 + eps-truncation PSNR)
    hist_<view>_<res>.npz raw per-pixel arrays at native resolution
    heat_ncontrib_<view>_<res>.png / heat_neval_<view>_<res>.png / render_<view>_<res>.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from gsplat.cuda._wrapper import rasterize_to_indices_in_range_2dgs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    composite,
    crop_K,
    eval_alpha,
    evaluated_counts,
    extract_contributors,
    load_cameras,
    load_splats,
    make_synthetic_scene,
    psnr,
    render_stages,
    save_png,
    scale_K,
    segment_weights,
)

STAT_NAMES = ("mean", "p50", "p90", "p99", "max")
PIXEL_METRICS = ("n_contrib", "n_contrib_eps", "n_eval", "tile_len", "acc_alpha", "ratio")
TIMING_COLUMNS = [
    "view", "res_w", "res_h", "N", "n_isects",
    "t_projection_ms", "t_tiling_sort_ms", "t_sh_ms", "t_blend_ms", "t_total_ms", "blend_frac",
]
PIXEL_COLUMNS = ["view", "res_w", "res_h", "N", "eps", "psnr_trunc_eps", "frac_saturated"] + [
    f"{m}_{s}" for m in PIXEL_METRICS for s in STAT_NAMES
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default=None, help="simple_trainer_2dgs checkpoint (.pt)")
    ap.add_argument("--data_dir", type=str, default=None, help="COLMAP dataset dir")
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--views", type=str, default=None, help="comma-separated camera indices (default 0,40; full run e.g. 0,20,40,60,80)")
    ap.add_argument(
        "--resolutions", type=str, default=None,
        help="comma-separated entries: 'native', explicit WxH, or a target width (height by aspect). "
             "Default native (synthetic: native,512); full sweep: native,1920,3840",
    )
    ap.add_argument("--eps", type=str, default="0.001,0.005,0.01,0.05", help="weight thresholds")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--synthetic", action="store_true", help="use common.make_synthetic_scene instead of a checkpoint")
    ap.add_argument("--seed", type=int, default=0, help="synthetic scene seed")
    ap.add_argument("--synthetic_sh", type=int, default=3, help="SH degree of the synthetic scene (-1 = plain RGB)")
    ap.add_argument("--tile_size", type=int, default=16)
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--n_iter", type=int, default=5)
    ap.add_argument("--crop_px", type=float, default=2.5e6, help="crop the A2 extraction when W*H exceeds this")
    ap.add_argument("--crop_grid", type=int, default=2, help="crop grid (2 -> 2x2 quadrants)")
    ap.add_argument(
        "--crop_mode", choices=("mask", "K"), default="mask",
        help="mask: full-frame projection, contributor extraction restricted per crop via a zeroed initial "
             "transmittance (bit-exact vs. un-cropped); K: re-project each crop with crop_K (PLAN route; "
             "float32 radius/means2d differences make a few pixels differ)",
    )
    ap.add_argument(
        "--force_crops", action="store_true",
        help="always use the crop path for A2; when the frame is small enough, also run the un-cropped "
             "extraction and check that the stitched per-pixel arrays match it",
    )
    ap.add_argument("--no_png", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Scene / view setup
# --------------------------------------------------------------------------- #
def build_views(args: argparse.Namespace, device) -> Tuple[dict, Optional[int], List[dict]]:
    """Returns ``(splats, sh_degree, views)`` where each view is a dict with
    ``id``, ``viewmat[4,4]`` tensor, ``K[3,3]`` tensor, ``W``, ``H`` (native)."""
    if args.synthetic:
        sh_degree = None if args.synthetic_sh < 0 else int(args.synthetic_sh)
        splats, viewmat, K, W, H = make_synthetic_scene(args.seed, sh_degree=sh_degree, device=device)
        view_ids = [0] if args.views is None else [int(v) for v in args.views.split(",")]
        views = [{"id": v, "viewmat": viewmat, "K": K, "W": W, "H": H} for v in view_ids]
        return splats, sh_degree, views

    if args.ckpt is None or args.data_dir is None:
        raise SystemExit("--ckpt and --data_dir are required unless --synthetic is given")
    splats = load_splats(args.ckpt, device)
    sh_degree = splats["sh_degree"]
    cams = load_cameras(args.data_dir, args.data_factor)
    view_ids = [0, 40] if args.views is None else [int(v) for v in args.views.split(",")]
    n_cams = cams["camtoworlds"].shape[0]
    views = []
    for v in view_ids:
        if v < 0 or v >= n_cams:
            raise SystemExit(f"view {v} out of range (dataset has {n_cams} cameras)")
        c2w = cams["camtoworlds"][v].astype(np.float64)
        viewmat = torch.as_tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
        cam_id = cams["camera_ids"][v]
        K = torch.as_tensor(cams["Ks_dict"][cam_id], dtype=torch.float32, device=device)
        W, H = cams["imsize_dict"][cam_id]
        views.append({"id": v, "viewmat": viewmat, "K": K, "W": int(W), "H": int(H)})
    return splats, sh_degree, views


def resolution_list(spec: str, W0: int, H0: int) -> List[Tuple[str, int, int]]:
    """``(label, W, H)`` per entry: 'native', an explicit ``WxH`` (the aspect
    ratio is NOT preserved — K is scaled per axis), or a target width whose
    height keeps the aspect ratio and is rounded to a multiple of 2."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "native":
            out.append(("native", W0, H0))
        elif "x" in tok.lower():
            w, h = (int(t) for t in tok.lower().split("x"))
            out.append((tok, w, h))
        else:
            w = int(tok)
            h = int(round(w * H0 / W0 / 2.0)) * 2
            out.append((tok, w, max(h, 2)))
    return out


def plan_crops(W: int, H: int, tile_size: int, grid: int, force: bool, crop_px: float) -> List[Tuple[int, int, int, int]]:
    """``(x0, y0, cw, ch)`` crops that exactly partition the frame. Split lines
    are snapped to multiples of ``tile_size`` so every crop tile is also a
    full-frame tile (keeps ``tile_len``/``n_eval`` comparable); the last crop
    takes the remainder."""
    if not force and W * H <= crop_px:
        return [(0, 0, W, H)]

    def split(n: int) -> List[Tuple[int, int]]:
        if n < 2 * tile_size or grid <= 1:
            return [(0, n)]
        bounds = [0]
        for k in range(1, grid):
            s = int(round(n * k / grid / tile_size)) * tile_size
            s = min(max(s, bounds[-1] + tile_size), n - tile_size)
            if s <= bounds[-1]:
                continue
            bounds.append(s)
        bounds.append(n)
        return [(bounds[i], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]

    crops = [(x0, y0, cw, ch) for (y0, ch) in split(H) for (x0, cw) in split(W)]
    assert sum(cw * ch for _, _, cw, ch in crops) == W * H, crops
    return crops


# --------------------------------------------------------------------------- #
# A2 extraction (optionally in crops)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_contributors_masked(stage: dict, W: int, H: int, x0: int, y0: int, cw: int, ch: int) -> dict:
    """Like :func:`common.extract_contributors` (same kernel, same full-frame
    projection / tile lists) but only for pixels inside the crop: the initial
    transmittance is 0 outside it, which makes the index kernel emit nothing
    for those pixels (``next_trans = 0 <= 1e-4`` on the first candidate).
    Pixel ids stay in full-frame row-major order, so the result is a bit-exact
    subset of the un-cropped extraction."""
    if (x0, y0, cw, ch) == (0, 0, W, H):
        return extract_contributors(stage, W, H)
    device = stage["means2d"].device
    trans = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    trans[0, y0 : y0 + ch, x0 : x0 + cw] = 1.0
    gs_ids, pix_ids, _ = rasterize_to_indices_in_range_2dgs(
        0, 1 << 30, trans,
        stage["means2d"], stage["ray_transforms"], stage["opacities"],
        W, H, stage["tile_size"], stage["isect_offsets"], stage["flatten_ids"],
    )
    gs_ids = gs_ids.to(torch.int64)
    pix_ids = pix_ids.to(torch.int64)
    if pix_ids.numel() > 1:
        assert bool((pix_ids[1:] >= pix_ids[:-1]).all().item()), "kernel output not grouped by pixel"
    ii = torch.div(pix_ids, W, rounding_mode="floor")
    jj = pix_ids % W
    assert bool(((ii >= y0) & (ii < y0 + ch) & (jj >= x0) & (jj < x0 + cw)).all().item()), "entries outside crop"
    n_contrib = torch.bincount(pix_ids, minlength=H * W)
    offsets = torch.zeros(H * W + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(n_contrib, dim=0)
    return {"gs_ids": gs_ids, "pix_ids": pix_ids, "offsets": offsets, "n_contrib": n_contrib}


@torch.no_grad()
def extract_view(
    splats: dict,
    viewmat: Tensor,
    K: Tensor,
    W: int,
    H: int,
    sh_degree: Optional[int],
    tile_size: int,
    eps_list: Sequence[float],
    crops: List[Tuple[int, int, int, int]],
    stage_full: Optional[dict] = None,
    keep_render: bool = True,
    crop_mode: str = "mask",
) -> dict:
    """Per-pixel A2 quantities for one view at ``(W, H)``, stitched to full-frame
    row-major order.

    ``crop_mode="mask"``: uses ``stage_full`` (rendered here if not given) for
    every crop and restricts the contributor list per crop via a masked initial
    transmittance. ``crop_mode="K"``: re-renders each crop with ``crop_K`` and
    works in crop-local pixel coordinates (``stage_full`` is only reused when
    there is a single crop)."""
    device = splats["means"].device
    P = W * H
    full = {
        "n_contrib": torch.zeros((H, W), dtype=torch.int64, device=device),
        "n_eval": torch.zeros((H, W), dtype=torch.int64, device=device),
        "tile_len": torch.zeros((H, W), dtype=torch.int64, device=device),
        "saturated": torch.zeros((H, W), dtype=torch.bool, device=device),
        "acc_alpha": torch.zeros((H, W), dtype=torch.float32, device=device),
    }
    n_contrib_eps = {eps: torch.zeros((H, W), dtype=torch.int64, device=device) for eps in eps_list}
    sse_eps = {eps: 0.0 for eps in eps_list}
    render = np.zeros((H, W, 3), dtype=np.float32) if keep_render else None
    M_total = 0
    use_full = crop_mode == "mask" or len(crops) == 1
    if use_full and stage_full is None:
        stage_full = render_stages(splats, viewmat, K, W, H, sh_degree=sh_degree, tile_size=tile_size)

    for ci, (x0, y0, cw, ch) in enumerate(crops):
        t0 = time.perf_counter()
        sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))  # full-frame destination
        if use_full:
            stage = stage_full
            ew, eh = W, H  # extraction pixel space = full frame
            lsl = sl  # where the crop lives inside the extraction space
            con = extract_contributors_masked(stage, W, H, x0, y0, cw, ch)
        else:
            stage = render_stages(
                splats, viewmat, crop_K(K, x0, y0), cw, ch, sh_degree=sh_degree, tile_size=tile_size
            )
            ew, eh = cw, ch
            lsl = (slice(0, ch), slice(0, cw))
            con = extract_contributors(stage, cw, ch)
        gs_ids, pix_ids, offsets = con["gs_ids"], con["pix_ids"], con["offsets"]
        M_total += int(gs_ids.numel())
        alpha = eval_alpha(stage, gs_ids, pix_ids, ew)
        _, w = segment_weights(alpha, offsets)
        del alpha
        n_eval, tile_len, saturated = evaluated_counts(
            stage, gs_ids, pix_ids, offsets, ew, eh, tile_size
        )
        rc = stage["render_colors"][0][lsl]  # [ch, cw, 3]
        full["n_contrib"][sl] = con["n_contrib"].reshape(eh, ew)[lsl]
        full["n_eval"][sl] = n_eval.reshape(eh, ew)[lsl]
        full["tile_len"][sl] = tile_len.reshape(eh, ew)[lsl]
        full["saturated"][sl] = saturated.reshape(eh, ew)[lsl]
        full["acc_alpha"][sl] = stage["render_alphas"][0, ..., 0][lsl]
        if render is not None:
            render[y0 : y0 + ch, x0 : x0 + cw] = rc.clamp(0, 1).cpu().numpy()
        for eps in eps_list:
            keep = w > eps
            n_contrib_eps[eps][sl] = torch.bincount(pix_ids[keep], minlength=ew * eh).reshape(eh, ew)[lsl]
            rgb_t, _ = composite(w[keep], gs_ids[keep], pix_ids[keep], stage["colors_rgb"], eh, ew)
            sse_eps[eps] += float(((rgb_t[lsl] - rc) ** 2).sum().item())
            del rgb_t, keep
        dt = time.perf_counter() - t0
        print(
            f"    crop {ci + 1}/{len(crops)} ({cw}x{ch} @ {x0},{y0}, {crop_mode}): M={gs_ids.numel()} "
            f"contrib/pix mean={full['n_contrib'][sl].float().mean():.2f} "
            f"n_eval mean={full['n_eval'][sl].float().mean():.1f} ({dt:.1f}s)"
        )
        del con, gs_ids, pix_ids, offsets, w, n_eval, tile_len, saturated, rc
        if stage is not stage_full:
            del stage
        torch.cuda.empty_cache()

    psnr_trunc = {}
    for eps in eps_list:
        mse = sse_eps[eps] / float(P * 3)
        psnr_trunc[eps] = float("inf") if mse <= 0 else float(10.0 * math.log10(1.0 / mse))
    full["n_contrib_eps"] = n_contrib_eps
    full["psnr_trunc"] = psnr_trunc
    full["render"] = render
    full["M"] = M_total
    return full


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p50, p90, p99 = np.percentile(x, [50.0, 90.0, 99.0])
    return {"mean": float(x.mean()), "p50": float(p50), "p90": float(p90), "p99": float(p99), "max": float(x.max())}


def pixel_rows(view_id: int, W: int, H: int, N: int, ex: dict, eps_list: Sequence[float]) -> List[dict]:
    n_contrib = ex["n_contrib"].cpu().numpy()
    n_eval = ex["n_eval"].cpu().numpy()
    tile_len = ex["tile_len"].cpu().numpy()
    acc_alpha = ex["acc_alpha"].cpu().numpy()
    frac_sat = float(ex["saturated"].float().mean().item())
    base_stats = {
        "n_contrib": _stats(n_contrib),
        "n_eval": _stats(n_eval),
        "tile_len": _stats(tile_len),
        "acc_alpha": _stats(acc_alpha),
    }
    rows = []
    for eps in eps_list:
        nce = ex["n_contrib_eps"][eps].cpu().numpy()
        ratio = n_eval.astype(np.float64) / np.maximum(nce, 1).astype(np.float64)
        row = {
            "view": view_id, "res_w": W, "res_h": H, "N": N, "eps": eps,
            "psnr_trunc_eps": ex["psnr_trunc"][eps], "frac_saturated": frac_sat,
        }
        stats = dict(base_stats)
        stats["n_contrib_eps"] = _stats(nce)
        stats["ratio"] = _stats(ratio)
        for m in PIXEL_METRICS:
            for s in STAT_NAMES:
                row[f"{m}_{s}"] = stats[m][s]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def write_csv(path: str, columns: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: _fmt(r.get(c, "")) for c in columns})


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.6g}"
    return str(v)


def heat_png(path: str, arr: np.ndarray, cmap_name: str = "inferno") -> None:
    """Heatmap normalized by the 99th percentile, through a matplotlib colormap."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    cmap = getattr(cm, cmap_name)
    x = np.asarray(arr, dtype=np.float32)
    p99 = float(np.percentile(x, 99.0))
    x = np.clip(x / p99, 0.0, 1.0) if p99 > 0 else np.zeros_like(x)
    save_png(path, cmap(x)[..., :3])


def save_hist(path: str, ex: dict, W: int, H: int) -> None:
    np.savez_compressed(
        path,
        W=np.int32(W), H=np.int32(H),
        n_contrib=ex["n_contrib"].cpu().numpy().astype(np.int32),
        n_eval=ex["n_eval"].cpu().numpy().astype(np.int32),
        tile_len=ex["tile_len"].cpu().numpy().astype(np.int32),
        acc_alpha=ex["acc_alpha"].cpu().numpy().astype(np.float16),
        saturated=ex["saturated"].cpu().numpy(),
        **{f"n_contrib_eps_{eps:g}": v.cpu().numpy().astype(np.int32) for eps, v in ex["n_contrib_eps"].items()},
    )


def check_crops(ex_crop: dict, ex_full: dict, eps_list: Sequence[float]) -> bool:
    """Compare stitched crop extraction against the un-cropped one."""
    ok = True
    for key in ("n_contrib", "n_eval", "tile_len", "saturated"):
        a, b = ex_crop[key], ex_full[key]
        mism = int((a != b).sum().item())
        maxd = int((a.long() - b.long()).abs().max().item()) if mism else 0
        flag = "OK" if mism == 0 else "MISMATCH"
        print(f"    [crop-check] {key:10s}: {mism} differing pixels (max |diff|={maxd}) {flag}")
        if key == "n_contrib" and mism:
            ok = False
    d_alpha = float((ex_crop["acc_alpha"] - ex_full["acc_alpha"]).abs().max().item())
    print(f"    [crop-check] acc_alpha : max |diff|={d_alpha:.2e}")
    for eps in eps_list:
        mism = int((ex_crop["n_contrib_eps"][eps] != ex_full["n_contrib_eps"][eps]).sum().item())
        print(f"    [crop-check] n_contrib_eps[{eps:g}]: {mism} differing pixels")
    if ex_crop["render"] is not None and ex_full["render"] is not None:
        print(f"    [crop-check] render    : max |diff|={np.abs(ex_crop['render'] - ex_full['render']).max():.2e}")
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    splats, sh_degree, views = build_views(args, device)
    N = int(splats["N"])
    eps_list = [float(e) for e in args.eps.split(",") if e.strip()]
    # Native frames only by default (speed on the laptop GPU); pass e.g.
    # --resolutions native,1920,3840 for the full-resolution sweep.
    res_spec = args.resolutions or ("native,512" if args.synthetic else "native")
    ts = int(args.tile_size)
    print(
        f"scene: N={N} sh_degree={sh_degree} | views={[v['id'] for v in views]} "
        f"| native={views[0]['W']}x{views[0]['H']} | resolutions={res_spec} | eps={eps_list} "
        f"| crop_mode={args.crop_mode} (crop when W*H > {args.crop_px:g})"
    )

    timing_rows: List[dict] = []
    pixel_rows_all: List[dict] = []
    first_view = views[0]["id"]

    for view in views:
        vid, viewmat, K0, W0, H0 = view["id"], view["viewmat"], view["K"], view["W"], view["H"]
        for label, W, H in resolution_list(res_spec, W0, H0):
            print(f"[view {vid}] res={label} ({W}x{H})")
            K = scale_K(K0, W0, H0, W, H)

            # ---- A1: full-frame timing --------------------------------------
            stage = render_stages(
                splats, viewmat, K, W, H, sh_degree=sh_degree, timing=True,
                n_warmup=args.n_warmup, n_iter=args.n_iter, tile_size=ts,
            )
            t = stage["times_ms"]
            stage_sum = t["projection"] + t["tiling_sort"] + t["sh"] + t["blend"]
            blend_frac = t["blend"] / stage_sum if stage_sum > 0 else float("nan")
            timing_rows.append({
                "view": vid, "res_w": W, "res_h": H, "N": N, "n_isects": stage["n_isects"],
                "t_projection_ms": t["projection"], "t_tiling_sort_ms": t["tiling_sort"],
                "t_sh_ms": t["sh"], "t_blend_ms": t["blend"], "t_total_ms": t["total"],
                "blend_frac": blend_frac,
            })
            print(
                f"    A1: n_isects={stage['n_isects']} proj={t['projection']:.2f} "
                f"tile+sort={t['tiling_sort']:.2f} sh={t['sh']:.2f} blend={t['blend']:.2f} "
                f"total={t['total']:.2f} ms | blend_frac={blend_frac:.3f}"
            )

            # ---- A2: contributors / evaluated ----------------------------------
            crops = plan_crops(W, H, ts, args.crop_grid, args.force_crops, args.crop_px)
            want_png = (not args.no_png) and vid == first_view and label == "native"
            ex_full = None
            if args.force_crops and W * H <= args.crop_px:
                print("    A2 (un-cropped reference for --force_crops check)")
                ex_full = extract_view(
                    splats, viewmat, K, W, H, sh_degree, ts, eps_list, [(0, 0, W, H)],
                    stage_full=stage, keep_render=True,
                )
            if len(crops) > 1:
                print(f"    A2 in {len(crops)} crops (mode={args.crop_mode}): {crops}")
                if args.crop_mode == "K":
                    del stage  # each crop is re-projected; free the full-frame stage first
                    stage = None
                    torch.cuda.empty_cache()
            ex = extract_view(
                splats, viewmat, K, W, H, sh_degree, ts, eps_list, crops,
                stage_full=stage, keep_render=want_png or ex_full is not None, crop_mode=args.crop_mode,
            )
            if ex_full is not None:
                check_crops(ex, ex_full, eps_list)
                del ex_full
            del stage
            torch.cuda.empty_cache()

            rows = pixel_rows(vid, W, H, N, ex, eps_list)
            pixel_rows_all.extend(rows)
            r0 = rows[0]
            print(
                f"    A2: M={ex['M']} n_contrib mean={r0['n_contrib_mean']:.2f} p50={r0['n_contrib_p50']:.0f} "
                f"max={r0['n_contrib_max']:.0f} | n_eval mean={r0['n_eval_mean']:.1f} p50={r0['n_eval_p50']:.0f} "
                f"| tile_len mean={r0['tile_len_mean']:.1f} | sat={r0['frac_saturated']:.3f}"
            )
            for r in rows:
                print(
                    f"        eps={r['eps']:g}: contrib_eps p50={r['n_contrib_eps_p50']:.0f} "
                    f"ratio p50={r['ratio_p50']:.2f} p90={r['ratio_p90']:.2f} "
                    f"| psnr_trunc={r['psnr_trunc_eps']:.2f} dB"
                )

            if label == "native":
                p = os.path.join(args.out, f"hist_{vid}_{label}.npz")
                save_hist(p, ex, W, H)
                print(f"    wrote {p}")
            if want_png:
                heat_png(os.path.join(args.out, f"heat_ncontrib_{vid}_{label}.png"), ex["n_contrib"].cpu().numpy())
                heat_png(os.path.join(args.out, f"heat_neval_{vid}_{label}.png"), ex["n_eval"].cpu().numpy())
                save_png(os.path.join(args.out, f"render_{vid}_{label}.png"), ex["render"])
                print(f"    wrote heat_ncontrib/heat_neval/render PNGs for view {vid} @ {label}")

            del ex, rows
            torch.cuda.empty_cache()

    p_t = os.path.join(args.out, "premise_timing.csv")
    p_p = os.path.join(args.out, "premise_pixels.csv")
    write_csv(p_t, TIMING_COLUMNS, timing_rows)
    write_csv(p_p, PIXEL_COLUMNS, pixel_rows_all)
    print(f"\nwrote {p_t} ({len(timing_rows)} rows)")
    print(f"wrote {p_p} ({len(pixel_rows_all)} rows)")

    # ---- summary ------------------------------------------------------------
    print("\n=== A1: median blend fraction per resolution ===")
    for (w, h) in sorted({(r["res_w"], r["res_h"]) for r in timing_rows}):
        bf = [r["blend_frac"] for r in timing_rows if (r["res_w"], r["res_h"]) == (w, h)]
        tt = [r["t_total_ms"] for r in timing_rows if (r["res_w"], r["res_h"]) == (w, h)]
        print(f"  {w}x{h}: blend_frac median={np.median(bf):.3f}  (total median {np.median(tt):.2f} ms, {len(bf)} views)")
    print("=== A2: median evaluated/contributing ratio (ratio_p50) per eps, across views ===")
    for eps in eps_list:
        sel = [r for r in pixel_rows_all if r["eps"] == eps]
        by_res = sorted({(r["res_w"], r["res_h"]) for r in sel})
        parts = []
        for (w, h) in by_res:
            v = [r["ratio_p50"] for r in sel if (r["res_w"], r["res_h"]) == (w, h)]
            parts.append(f"{w}x{h}: {np.median(v):.2f}")
        allv = [r["ratio_p50"] for r in sel]
        print(f"  eps={eps:g}: all={np.median(allv):.2f} | " + ", ".join(parts))
    print(f"\noutputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
