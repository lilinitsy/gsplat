#!/usr/bin/env python
"""Phase 0 / WP-D: turn the measurement CSVs into ``report.md`` + plots.

Reads (whichever exist) from ``--out``:

* ``premise_timing.csv``  (WP-B, A1)  per view x resolution stage timings
* ``premise_pixels.csv``  (WP-B, A2)  per view x resolution x eps pixel statistics
* ``temporal_pairs.csv``  (WP-C, B1-B5, C1) per pose pair temporal statistics

Writes ``report.md``, ``plot_stages.png``, ``plot_iou.png``, ``plot_oracle.png``,
``plot_fallback.png`` into the same directory, and prints the go/no-go block.

A missing CSV never crashes the report: its sections are omitted and the
go/no-go criteria that depend on it are marked N/A.

``--demo`` writes small synthetic CSVs (same column names) into a temporary
directory and runs the report on them, for developing/testing this script
without a trained scene.

Only the standard library + matplotlib (Agg) are used; no numpy/pandas needed.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
import tempfile
from typing import Any, Callable, Iterable, Sequence

import matplotlib

# Windows consoles default to cp1252, which cannot encode the ≥/ε/τ used below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------------------- #
# Thresholds (from the table at the top of PLAN.md)
# --------------------------------------------------------------------------- #
THRESH_A1_BLEND_FRAC = 0.50  # blend >= 50 % of frame time at the largest resolution
THRESH_A2_RATIO = 5.0  # median evaluated / contributing >= 5x
THRESH_B1_IOU = 0.90  # median IoU > 0.9 at one-frame deltas
THRESH_B2_PSNR = 35.0  # oracle PSNR >= 35 dB at one-frame deltas
THRESH_B3_FALLBACK = 0.05  # fallback fraction < 5 % at one-frame deltas
LPIPS_IMPERCEPTIBLE = 0.05  # informational only ("imperceptible"); not a gate

# "One-frame" bands from PLAN.md, drawn on the plots as context.
ONE_FRAME_ROT_DEG = (0.25, 1.0)
ONE_FRAME_TRANS_PCT = (0.5, 2.0)

# --------------------------------------------------------------------------- #
# Styling (reference palette; fixed categorical order, never cycled)
# --------------------------------------------------------------------------- #
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f0efec"

STAGES = [("t_projection_ms", "projection"), ("t_tiling_sort_ms", "tiling + sort"), ("t_sh_ms", "SH"), ("t_blend_ms", "blend")]
STAGE_COLOR = {name: PALETTE[i] for i, (_, name) in enumerate(STAGES)}

KINDS_ROT = ["rot_yaw", "rot_pitch"]
KINDS_TRANS = ["trans_x", "trans_z"]
KIND_COLOR = {"rot_yaw": PALETTE[0], "rot_pitch": PALETTE[1], "trans_x": PALETTE[2], "trans_z": PALETTE[3], "traj": PALETTE[4]}
KIND_LABEL = {"rot_yaw": "yaw", "rot_pitch": "pitch", "trans_x": "translate x", "trans_z": "translate z", "traj": "trajectory"}

NAN = float("nan")


def style_axes(ax: plt.Axes, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def new_figure(nrows: int, ncols: int, size: tuple[float, float]) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def legend(ax: plt.Axes, handles: Sequence[Any] | None = None, **kw: Any) -> None:
    lg = ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, **kw) if handles else ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)
    if lg is not None:
        lg.get_frame().set_facecolor(SURFACE)


# --------------------------------------------------------------------------- #
# CSV helpers (no pandas)
# --------------------------------------------------------------------------- #
Row = dict[str, str]


def read_csv(path: str) -> list[Row] | None:
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def fnum(row: Row, key: str) -> float:
    v = row.get(key)
    if v is None:
        return NAN
    try:
        return float(v)
    except ValueError:
        return NAN


def finite(vals: Iterable[float]) -> list[float]:
    return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]


def median(vals: Iterable[float]) -> float:
    xs = sorted(finite(vals))
    if not xs:
        return NAN
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(vals: Iterable[float]) -> float:
    xs = finite(vals)
    return sum(xs) / len(xs) if xs else NAN


def fmt(x: float, nd: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "–"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return "–" if not math.isfinite(x) else f"{100 * x:.{nd}f} %"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def group_by(rows: Iterable[Row], keyfn: Callable[[Row], Any]) -> dict[Any, list[Row]]:
    groups: dict[Any, list[Row]] = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    return groups


def median_over(rows: Sequence[Row], col: str) -> float:
    return median(fnum(r, col) for r in rows)


def res_key(r: Row) -> tuple[int, int]:
    return int(fnum(r, "res_w")), int(fnum(r, "res_h"))


def res_label(wh: tuple[int, int]) -> str:
    return f"{wh[0]}×{wh[1]}"


# --------------------------------------------------------------------------- #
# A1: stage timings
# --------------------------------------------------------------------------- #
def section_timing(rows: list[Row], out_dir: str) -> tuple[str, dict[str, Any]]:
    by_res = group_by(rows, res_key)
    res_list = sorted(by_res)
    agg: dict[tuple[int, int], dict[str, float]] = {}
    for wh in res_list:
        rs = by_res[wh]
        d = {col: median_over(rs, col) for col, _ in STAGES}
        d["t_total_ms"] = median_over(rs, "t_total_ms")
        d["blend_frac"] = median_over(rs, "blend_frac")
        d["N"] = median_over(rs, "N")
        d["n_isects"] = median_over(rs, "n_isects")
        d["n_views"] = len(rs)
        agg[wh] = d

    # ---- plot: stacked bars of stage time per resolution
    fig, axes = new_figure(1, 1, (7.0, 4.2))
    ax = axes[0][0]
    xs = list(range(len(res_list)))
    bottoms = [0.0] * len(res_list)
    for col, name in STAGES:
        vals = [agg[wh][col] if math.isfinite(agg[wh][col]) else 0.0 for wh in res_list]
        ax.bar(xs, vals, bottom=bottoms, width=0.55, color=STAGE_COLOR[name], edgecolor=SURFACE, linewidth=2, label=name)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, wh in zip(xs, res_list):
        bf = agg[wh]["blend_frac"]
        if math.isfinite(bf):
            ax.text(x, bottoms[x] * 1.02 + 0.3, f"blend {100 * bf:.0f} %", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([res_label(wh) for wh in res_list], color=INK2)
    ax.set_ylim(0, max(bottoms + [1.0]) * 1.18)
    style_axes(ax, title="A1 — per-stage render time (median over views)", ylabel="ms per frame")
    legend(ax, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_stages.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- table
    table_rows = []
    for wh in res_list:
        d = agg[wh]
        table_rows.append([res_label(wh), d["n_views"], fmt(d["N"], 0), fmt(d["n_isects"], 0)] + [fmt(d[c], 2) for c, _ in STAGES] + [fmt(d["t_total_ms"], 2), pct(d["blend_frac"])])
    md = ["## A1 — Per-stage timing", "", f"Rows: {len(rows)} (views × resolutions). Values are medians over views at each resolution.", "", md_table(["resolution", "views", "N", "isects"] + [n for _, n in STAGES] + ["total ms", "blend frac"], table_rows), "", "![stage times](plot_stages.png)", ""]
    largest = res_list[-1] if res_list else None
    return "\n".join(md), {"largest_res": largest, "blend_frac_largest": agg[largest]["blend_frac"] if largest else NAN}


# --------------------------------------------------------------------------- #
# A2: per-pixel statistics
# --------------------------------------------------------------------------- #
STAT_NAMES = ["mean", "p50", "p90", "p99", "max"]
STAT_RE = re.compile(r"^(?P<base>.+)_(?P<stat>mean|p50|p90|p99|max)$")


def detect_pixel_metrics(fieldnames: Iterable[str]) -> dict[str, dict[str, str]]:
    """Map metric base name -> {stat: column} from ``<base>_<stat>`` columns."""
    metrics: dict[str, dict[str, str]] = {}
    for col in fieldnames:
        m = STAT_RE.match(col)
        if m:
            metrics.setdefault(m.group("base"), {})[m.group("stat")] = col
    return metrics


def pick_ratio_metric(metrics: dict[str, dict[str, str]]) -> str | None:
    """Find the evaluated/contributing ratio metric, whatever WP-B called it."""
    for base in metrics:
        b = base.lower()
        if "ratio" in b or ("eval" in b and "contrib" in b) or "over" in b:
            return base
    return None


def pick_metric(metrics: dict[str, dict[str, str]], *candidates: str) -> str | None:
    for c in candidates:
        if c in metrics:
            return c
    return None


def section_pixels(rows: list[Row], out_dir: str) -> tuple[str, dict[str, Any]]:
    fields = list(rows[0].keys()) if rows else []
    metrics = detect_pixel_metrics(fields)
    m_ratio = pick_ratio_metric(metrics)
    m_contrib = pick_metric(metrics, "n_contrib", "contrib")
    m_contrib_eps = pick_metric(metrics, "n_contrib_eps", "contrib_eps", "n_contrib_w")
    m_eval = pick_metric(metrics, "n_eval", "eval")
    m_tile = pick_metric(metrics, "tile_len")
    m_alpha = pick_metric(metrics, "acc_alpha", "alpha")
    have_psnr = "psnr_trunc_eps" in fields

    groups = group_by(rows, lambda r: (res_key(r), fnum(r, "eps")))
    keys = sorted(groups)
    res_list = sorted({k[0] for k in keys})
    eps_list = sorted({k[1] for k in keys})

    def stat(rs: Sequence[Row], base: str | None, s: str) -> float:
        if base is None or s not in metrics.get(base, {}):
            return NAN
        return median_over(rs, metrics[base][s])

    def ratio_p50(rs: Sequence[Row]) -> tuple[float, bool]:
        """(median ratio, derived?)  Derived = p50(n_eval) / max(p50(n_contrib_eps), 1)."""
        if m_ratio is not None:
            v = stat(rs, m_ratio, "p50")
            if math.isfinite(v):
                return v, False
        ne, nc = stat(rs, m_eval, "p50"), stat(rs, m_contrib_eps, "p50")
        if math.isfinite(ne) and math.isfinite(nc):
            return ne / max(nc, 1.0), True
        return NAN, True

    derived_any = False
    table_rows = []
    for wh, eps in keys:
        rs = groups[(wh, eps)]
        rp50, derived = ratio_p50(rs)
        derived_any |= derived
        table_rows.append([res_label(wh), f"{eps:g}", len(rs), fmt(stat(rs, m_contrib, "p50"), 1), fmt(stat(rs, m_contrib, "p90"), 1), fmt(stat(rs, m_contrib_eps, "p50"), 1), fmt(stat(rs, m_contrib_eps, "p90"), 1), fmt(stat(rs, m_eval, "p50"), 1), fmt(stat(rs, m_eval, "p90"), 1), fmt(stat(rs, m_tile, "p50"), 1), fmt(stat(rs, m_alpha, "p50"), 3), fmt(stat(rs, m_ratio, "mean"), 2), fmt(rp50, 2) + ("*" if derived else ""), fmt(stat(rs, m_ratio, "p90"), 2), fmt(stat(rs, m_ratio, "p99"), 2), fmt(median_over(rs, "psnr_trunc_eps"), 2) if have_psnr else "–"])

    md = ["## A2 — Evaluated vs contributing surfels per pixel", "", f"Rows: {len(rows)} (views × resolutions × ε). Values are medians over views. `n_contrib` = raw kernel contributor count (α ≥ 1/255), `n_contrib_ε` = contributors with weight > ε, `n_eval` = surfels evaluated by the rasterizer for that pixel, `ratio` = n_eval / max(n_contrib_ε, 1). `psnr_trunc` = PSNR of the ε-truncated recomposite vs the full render.", ""]
    md.append(md_table(["resolution", "ε", "views", "contrib p50", "contrib p90", "contrib_ε p50", "contrib_ε p90", "eval p50", "eval p90", "tile_len p50", "acc_α p50", "ratio mean", "ratio p50", "ratio p90", "ratio p99", "psnr_trunc"], table_rows))
    if derived_any:
        md.append("\n\\* ratio p50 derived as p50(n_eval) / max(p50(n_contrib_ε), 1) because no per-pixel ratio column was found (approximate: ratio of medians, not median of ratios).\n")
    if m_ratio is None:
        md.append(f"\nDetected percentile metrics: {', '.join(sorted(metrics)) or 'none'}; no ratio metric among them.\n")

    # Full percentile table at the native (smallest) resolution.
    if res_list:
        native = res_list[0]
        md.append(f"\n### A2 — full percentiles at native resolution ({res_label(native)})\n")
        full_rows = []
        for eps in eps_list:
            rs = groups.get((native, eps), [])
            if not rs:
                continue
            for base in sorted(metrics):
                full_rows.append([f"{eps:g}", base] + [fmt(stat(rs, base, s), 2) for s in STAT_NAMES])
        md.append(md_table(["ε", "metric"] + STAT_NAMES, full_rows))

    # Go/no-go input: largest resolution, smallest ε (the most conservative cell).
    info: dict[str, Any] = {"res": None, "eps": None, "ratio_p50": NAN, "derived": False}
    if res_list and eps_list:
        rs = groups.get((res_list[-1], eps_list[0]), [])
        if rs:
            rp, derived = ratio_p50(rs)
            info = {"res": res_list[-1], "eps": eps_list[0], "ratio_p50": rp, "derived": derived}
    return "\n".join(md), info


# --------------------------------------------------------------------------- #
# Temporal (B1-B5, C1)
# --------------------------------------------------------------------------- #
TEMPORAL_COLS = ["rot_deg", "trans_frac", "N", "valid_frac", "iou_mean", "iou_p10", "iou_p50", "cap_bind_frac", "psnr_oracle", "lpips_oracle", "psnr_oracle_valid", "fallback_frac_tau0.01", "fallback_frac_tau0.05", "inversion_frac", "inv_depth_gap_p50", "psnr_stale_order", "union_frac", "quad_share", "psnr_stale_sh", "mean_candidates_per_pixel"]


def fallback_cols(rows: list[Row]) -> list[str]:
    cols = [c for c in (rows[0].keys() if rows else []) if c.startswith("fallback_frac_tau")]

    def tau(c: str) -> float:
        try:
            return float(c[len("fallback_frac_tau"):])
        except ValueError:
            return math.inf

    return sorted(cols, key=tau)


def aggregate_temporal(rows: list[Row]) -> dict[tuple[str, float], dict[str, float]]:
    """(kind, delta) -> median over views of every numeric column."""
    groups = group_by(rows, lambda r: (r.get("kind", "?"), fnum(r, "delta")))
    cols = set(TEMPORAL_COLS) | set(fallback_cols(rows))
    agg: dict[tuple[str, float], dict[str, float]] = {}
    for k, rs in groups.items():
        d = {c: median_over(rs, c) for c in cols}
        d["n_views"] = float(len(rs))
        agg[k] = d
    return agg


def kinds_present(agg: dict[tuple[str, float], dict[str, float]], wanted: Sequence[str]) -> list[str]:
    present = {k for k, _ in agg}
    return [k for k in wanted if k in present]


def series(agg: dict[tuple[str, float], dict[str, float]], kind: str, xcol: str, ycol: str, xscale: float = 1.0) -> tuple[list[float], list[float]]:
    pts = sorted((agg[(k, d)][xcol] * xscale, agg[(k, d)][ycol]) for (k, d) in agg if k == kind)
    pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
    return [p[0] for p in pts], [p[1] for p in pts]


def draw_band(ax: plt.Axes, lo: float, hi: float) -> None:
    ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)


def plot_pair(axes_row: Sequence[plt.Axes], agg: dict, ycols: Sequence[tuple[str, str, str]], *, title: str, ylabel: str, hline: float | None = None, hline_label: str = "", yscale: float = 1.0, ylim: tuple[float, float] | None = None) -> None:
    """Left panel: rotation kinds vs rot_deg; right panel: translation kinds vs trans %.

    ycols: list of (column, linestyle, label-suffix)."""
    specs = [(axes_row[0], KINDS_ROT, "rot_deg", 1.0, "rotation delta (deg)", ONE_FRAME_ROT_DEG), (axes_row[1], KINDS_TRANS, "trans_frac", 100.0, "translation delta (% of scene scale)", ONE_FRAME_TRANS_PCT)]
    for ax, kinds, xcol, xscale, xlabel, band in specs:
        draw_band(ax, *band)
        handles: list[Any] = []
        for kind in kinds_present(agg, kinds):
            for ycol, ls, suffix in ycols:
                x, y = series(agg, kind, xcol, ycol, xscale)
                if not x:
                    continue
                y = [v * yscale for v in y]
                (ln,) = ax.plot(x, y, ls, color=KIND_COLOR[kind], lw=2, marker="o", ms=5, mec=SURFACE, mew=1, label=f"{KIND_LABEL[kind]}{suffix}")
                handles.append(ln)
        ax.set_xscale("log")
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=1, ls=(0, (4, 3)))
            # x in axes fraction, y in data units: stays inside the axes on a log x-axis.
            ax.text(0.01, hline, hline_label, color=MUTED, fontsize=8, va="bottom", ha="left", transform=ax.get_yaxis_transform())
        if ylim is not None:
            ax.set_ylim(*ylim)
        style_axes(ax, title=title + (" — rotation" if xcol == "rot_deg" else " — translation"), xlabel=xlabel, ylabel=ylabel)
        handles.append(Patch(facecolor=BAND, label="one-frame band"))
        legend(ax, handles=handles, loc="best")


def section_temporal(rows: list[Row], out_dir: str) -> tuple[str, dict[str, Any]]:
    agg = aggregate_temporal(rows)
    fb_cols = fallback_cols(rows)
    md: list[str] = []

    # ---- B1 IoU
    fig, axes = new_figure(1, 2, (11, 4.2))
    plot_pair(axes[0], agg, [("iou_p50", "-", " p50"), ("iou_p10", "--", " p10")], title="B1 — contributing-set IoU", ylabel="IoU", hline=THRESH_B1_IOU, hline_label=f"threshold {THRESH_B1_IOU}", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_iou.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- B2 oracle PSNR / LPIPS + stale variants
    fig, axes = new_figure(3, 2, (11, 11.5))
    plot_pair(axes[0], agg, [("psnr_oracle", "-", " (all px)"), ("psnr_oracle_valid", "--", " (valid px)")], title="B2 — oracle warp PSNR", ylabel="PSNR (dB)", hline=THRESH_B2_PSNR, hline_label=f"threshold {THRESH_B2_PSNR:.0f} dB")
    plot_pair(axes[1], agg, [("lpips_oracle", "-", "")], title="B2 — oracle warp LPIPS (alex)", ylabel="LPIPS (lower is better)", hline=LPIPS_IMPERCEPTIBLE, hline_label="~imperceptible")
    plot_pair(axes[2], agg, [("psnr_stale_order", "-", " stale order (B4)"), ("psnr_stale_sh", "--", " stale SH (C1)")], title="B4 / C1 — stale-order and stale-SH PSNR", ylabel="PSNR (dB)", hline=THRESH_B2_PSNR, hline_label=f"{THRESH_B2_PSNR:.0f} dB")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_oracle.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- B3 fallback
    fig, axes = new_figure(1, 2, (11, 4.2))
    styles = ["-", "--", ":", "-."]
    ycols = [(c, styles[i % len(styles)], f" τ={c[len('fallback_frac_tau'):]}") for i, c in enumerate(fb_cols)]
    plot_pair(axes[0], agg, ycols, title="B3 — fallback fraction", ylabel="fallback pixels (%)", hline=100 * THRESH_B3_FALLBACK, hline_label=f"threshold {100 * THRESH_B3_FALLBACK:.0f} %", yscale=100.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_fallback.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- tables
    def row_keys(kinds: Sequence[str]) -> list[tuple[str, float]]:
        return sorted((k for k in agg if k[0] in kinds), key=lambda k: (kinds.index(k[0]), k[1]))

    def delta_str(k: tuple[str, float]) -> str:
        d = agg[k]
        if k[0].startswith("rot"):
            return f"{k[1]:g}° (meas. {fmt(d['rot_deg'], 2)}°)"
        if k[0].startswith("trans"):
            return f"{k[1]:g} % (meas. {fmt(100 * d['trans_frac'], 2)} %)"
        return f"step {k[1]:g} ({fmt(d['rot_deg'], 2)}°, {fmt(100 * d['trans_frac'], 2)} %)"

    N = median_over(rows, "N")
    eps_v = sorted({r.get("eps", "?") for r in rows})
    K_v = sorted({r.get("K", "?") for r in rows})
    r_v = sorted({r.get("radius", "?") for r in rows})
    md += ["## Temporal measurements (B1–B5, C1)", "", f"Rows: {len(rows)} pose pairs; N = {fmt(N, 0)}; ε = {', '.join(eps_v)}; K = {', '.join(K_v)}; radius = {', '.join(r_v)}. Values are medians over base views at each (kind, delta). The grey band on the plots marks the one-frame regime from PLAN.md ({ONE_FRAME_ROT_DEG[0]}–{ONE_FRAME_ROT_DEG[1]}°, {ONE_FRAME_TRANS_PCT[0]}–{ONE_FRAME_TRANS_PCT[1]} % of scene scale).", ""]

    sweep_keys = row_keys(KINDS_ROT + KINDS_TRANS)
    traj_keys = row_keys(["traj"])

    md += ["### B1 — Contributing-set IoU vs delta", "", "![IoU](plot_iou.png)", ""]
    md.append(md_table(["kind", "delta", "views", "valid frac", "IoU mean", "IoU p10", "IoU p50"], [[k[0], delta_str(k), int(agg[k]["n_views"]), pct(agg[k]["valid_frac"]), fmt(agg[k]["iou_mean"]), fmt(agg[k]["iou_p10"]), fmt(agg[k]["iou_p50"])] for k in sweep_keys]))

    md += ["", "### B2 — Oracle warp quality vs delta", "", "![oracle](plot_oracle.png)", ""]
    md.append(md_table(["kind", "delta", "PSNR oracle", "PSNR oracle (valid px)", "LPIPS oracle", "cap-bind frac", "mean candidates/px"], [[k[0], delta_str(k), fmt(agg[k]["psnr_oracle"], 2), fmt(agg[k]["psnr_oracle_valid"], 2), fmt(agg[k]["lpips_oracle"], 4), pct(agg[k]["cap_bind_frac"]), fmt(agg[k]["mean_candidates_per_pixel"], 1)] for k in sweep_keys]))

    md += ["", "### B3 — Fallback fraction vs delta", "", "![fallback](plot_fallback.png)", ""]
    md.append(md_table(["kind", "delta"] + [f"fallback τ={c[len('fallback_frac_tau'):]}" for c in fb_cols], [[k[0], delta_str(k)] + [pct(agg[k][c], 2) for c in fb_cols] for k in sweep_keys]))

    md += ["", "### B4 — Order stability (shared candidates, A's order vs B's order)", ""]
    md.append(md_table(["kind", "delta", "inversion frac", "inverted-pair A-depth gap p50", "PSNR stale order", "PSNR fresh order (oracle)"], [[k[0], delta_str(k), pct(agg[k]["inversion_frac"], 2), fmt(agg[k]["inv_depth_gap_p50"], 4), fmt(agg[k]["psnr_stale_order"], 2), fmt(agg[k]["psnr_oracle"], 2)] for k in sweep_keys]))

    md += ["", "### B5 — Candidate union size and 2×2-quad sharing", ""]
    md.append(md_table(["kind", "delta", "union |∪C(p)| / N", "quad share", "mean candidates/px"], [[k[0], delta_str(k), pct(agg[k]["union_frac"], 2), fmt(agg[k]["quad_share"], 3), fmt(agg[k]["mean_candidates_per_pixel"], 1)] for k in sweep_keys]))

    md += ["", "### C1 — Oracle warp with stale per-surfel RGB (no SH re-eval)", ""]
    md.append(md_table(["kind", "delta", "PSNR stale SH", "PSNR fresh SH (oracle)", "Δ dB"], [[k[0], delta_str(k), fmt(agg[k]["psnr_stale_sh"], 2), fmt(agg[k]["psnr_oracle"], 2), fmt(agg[k]["psnr_stale_sh"] - agg[k]["psnr_oracle"], 2)] for k in sweep_keys]))

    if traj_keys:
        md += ["", "### Trajectory pairs (interpolated path, consecutive frames)", ""]
        md.append(md_table(["step", "rot (°)", "trans (%)", "IoU p50", "PSNR oracle", "LPIPS oracle"] + [f"fallback τ={c[len('fallback_frac_tau'):]}" for c in fb_cols] + ["inversion frac", "PSNR stale order", "PSNR stale SH"], [[f"{k[1]:g}", fmt(agg[k]["rot_deg"], 3), fmt(100 * agg[k]["trans_frac"], 3), fmt(agg[k]["iou_p50"]), fmt(agg[k]["psnr_oracle"], 2), fmt(agg[k]["lpips_oracle"], 4)] + [pct(agg[k][c], 2) for c in fb_cols] + [pct(agg[k]["inversion_frac"], 2), fmt(agg[k]["psnr_stale_order"], 2), fmt(agg[k]["psnr_stale_sh"], 2)] for k in traj_keys]))

    # ---- go/no-go inputs: smallest rotation delta and smallest translation delta (median over kinds & views)
    def smallest(kinds: Sequence[str]) -> tuple[float, dict[str, float]] | None:
        ks = [k for k in agg if k[0] in kinds]
        if not ks:
            return None
        dmin = min(k[1] for k in ks)
        sel = [agg[k] for k in ks if k[1] == dmin]
        cols = set().union(*(d.keys() for d in sel))
        return dmin, {c: median(d.get(c, NAN) for d in sel) for c in cols}

    return "\n".join(md), {"rot": smallest(KINDS_ROT), "trans": smallest(KINDS_TRANS), "fb_cols": fb_cols}


# --------------------------------------------------------------------------- #
# Go / no-go
# --------------------------------------------------------------------------- #
def gonogo(timing: dict[str, Any] | None, pixels: dict[str, Any] | None, temporal: dict[str, Any] | None) -> tuple[str, list[str]]:
    lines: list[tuple[str, str, str, str]] = []  # (id, criterion, evidence, verdict)

    def verdict(ok: bool | None) -> str:
        return "N/A" if ok is None else ("PASS" if ok else "FAIL")

    # A1
    if timing and timing.get("largest_res"):
        bf = timing["blend_frac_largest"]
        ok = bf >= THRESH_A1_BLEND_FRAC if math.isfinite(bf) else None
        lines.append(("A1", f"blend ≥ {100 * THRESH_A1_BLEND_FRAC:.0f} % of frame time at the largest resolution", f"{res_label(timing['largest_res'])}: blend = {pct(bf)}", verdict(ok)))
    else:
        lines.append(("A1", f"blend ≥ {100 * THRESH_A1_BLEND_FRAC:.0f} % at the largest resolution", "premise_timing.csv missing", "N/A"))
    # A2
    if pixels and pixels.get("res"):
        rp = pixels["ratio_p50"]
        ok = rp >= THRESH_A2_RATIO if math.isfinite(rp) else None
        lines.append(("A2", f"median evaluated / contributing ≥ {THRESH_A2_RATIO:g}×", f"{res_label(pixels['res'])}, ε={pixels['eps']:g}: ratio p50 = {fmt(rp, 2)}{' (derived from medians)' if pixels['derived'] else ''}", verdict(ok)))
    else:
        lines.append(("A2", f"median evaluated / contributing ≥ {THRESH_A2_RATIO:g}×", "premise_pixels.csv missing", "N/A"))
    # B1-B3 at the smallest rotation and translation deltas
    if temporal and (temporal.get("rot") or temporal.get("trans")):
        rot, trans = temporal.get("rot"), temporal.get("trans")
        fb_cols = temporal.get("fb_cols") or []
        fb_col = fb_cols[0] if fb_cols else None  # smallest τ = most conservative

        def both(col: str, test: Callable[[float], bool], label: Callable[[float], str]) -> tuple[str, bool | None]:
            parts, oks = [], []
            for name, sel, unit in (("rot", rot, "°"), ("trans", trans, " %")):
                if sel is None:
                    parts.append(f"{name}: no rows")
                    continue
                dmin, d = sel
                v = d.get(col, NAN)
                parts.append(f"{name} {dmin:g}{unit}: {label(v)}")
                oks.append(test(v) if math.isfinite(v) else None)
            ok: bool | None = None if (not oks or any(o is None for o in oks)) else all(bool(o) for o in oks)
            return "; ".join(parts), ok

        ev, ok = both("iou_p50", lambda v: v > THRESH_B1_IOU, lambda v: f"IoU p50 = {fmt(v)}")
        lines.append(("B1", f"median IoU > {THRESH_B1_IOU} at the smallest deltas", ev, verdict(ok)))
        ev, ok = both("psnr_oracle", lambda v: v >= THRESH_B2_PSNR, lambda v: f"PSNR = {fmt(v, 2)} dB")
        ev2, _ = both("lpips_oracle", lambda v: v <= LPIPS_IMPERCEPTIBLE, lambda v: f"LPIPS = {fmt(v, 4)}")
        lines.append(("B2", f"oracle PSNR ≥ {THRESH_B2_PSNR:.0f} dB at the smallest deltas (LPIPS reported, not gated)", ev + " | " + ev2, verdict(ok)))
        if fb_col:
            ev, ok = both(fb_col, lambda v: v < THRESH_B3_FALLBACK, lambda v: f"fallback = {pct(v, 2)}")
            lines.append(("B3", f"fallback fraction < {100 * THRESH_B3_FALLBACK:.0f} % at the smallest deltas (τ={fb_col[len('fallback_frac_tau'):]})", ev, verdict(ok)))
        else:
            lines.append(("B3", f"fallback fraction < {100 * THRESH_B3_FALLBACK:.0f} %", "no fallback_frac_tau* column", "N/A"))
        ev, _ = both("inversion_frac", lambda v: True, lambda v: f"inversion frac = {pct(v, 2)}")
        ev2, _ = both("psnr_stale_order", lambda v: True, lambda v: f"PSNR stale order = {fmt(v, 2)} dB")
        lines.append(("B4", "order inversions / stale-order error (report only)", ev + " | " + ev2, "INFO"))
        ev, _ = both("union_frac", lambda v: True, lambda v: f"union = {pct(v, 2)} of N")
        ev2, _ = both("quad_share", lambda v: True, lambda v: f"quad share = {fmt(v)}")
        lines.append(("B5", "candidate union / quad sharing (report only)", ev + " | " + ev2, "INFO"))
        ev, _ = both("psnr_stale_sh", lambda v: True, lambda v: f"PSNR stale SH = {fmt(v, 2)} dB")
        lines.append(("C1", "stale-SH oracle quality (report only)", ev, "INFO"))
    else:
        for cid, crit in (("B1", f"median IoU > {THRESH_B1_IOU}"), ("B2", f"oracle PSNR ≥ {THRESH_B2_PSNR:.0f} dB"), ("B3", f"fallback < {100 * THRESH_B3_FALLBACK:.0f} %"), ("B4", "order inversions (report)"), ("B5", "union / quad share (report)"), ("C1", "stale SH (report)")):
            lines.append((cid, crit, "temporal_pairs.csv missing", "N/A"))

    gated = [l for l in lines if l[3] in ("PASS", "FAIL")]
    n_pass = sum(1 for l in gated if l[3] == "PASS")
    overall = "GO" if gated and n_pass == len(gated) else ("NO-GO" if any(l[3] == "FAIL" for l in gated) else "UNDETERMINED")
    md = ["## Go / no-go", "", f"Temporal criteria use the **smallest** rotation and translation deltas in the sweep (median over kinds and views); the timing criterion uses the **largest** resolution; A2 uses the largest resolution and smallest ε.", "", md_table(["ID", "criterion", "evidence", "verdict"], [list(l) for l in lines]), "", f"**Overall: {overall}** ({n_pass}/{len(gated)} gated criteria pass; {sum(1 for l in lines if l[3] == 'N/A')} N/A).", ""]
    printed = ["", "=== Go / no-go ==="] + [f"[{l[3]:>4}] {l[0]}  {l[1]}\n        {l[2]}" for l in lines] + [f"Overall: {overall} ({n_pass}/{len(gated)} gated criteria pass)"]
    return "\n".join(md), printed


# --------------------------------------------------------------------------- #
# Report driver
# --------------------------------------------------------------------------- #
def build_report(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    paths = {name: os.path.join(out_dir, f"{name}.csv") for name in ("premise_timing", "premise_pixels", "temporal_pairs")}
    data = {name: read_csv(p) for name, p in paths.items()}

    sections: list[str] = []
    missing: list[str] = []
    timing_info = pixels_info = temporal_info = None

    rows = data["premise_timing"]
    if rows:
        md, timing_info = section_timing(rows, out_dir)
        sections.append(md)
    else:
        missing.append("premise_timing.csv")
        sections.append("## A1 — Per-stage timing\n\n_Skipped: `premise_timing.csv` not found (or empty). Run `measure_premise.py`._\n")

    rows = data["premise_pixels"]
    if rows:
        md, pixels_info = section_pixels(rows, out_dir)
        sections.append(md)
    else:
        missing.append("premise_pixels.csv")
        sections.append("## A2 — Evaluated vs contributing surfels per pixel\n\n_Skipped: `premise_pixels.csv` not found (or empty). Run `measure_premise.py`._\n")

    rows = data["temporal_pairs"]
    if rows:
        md, temporal_info = section_temporal(rows, out_dir)
        sections.append(md)
    else:
        missing.append("temporal_pairs.csv")
        sections.append("## Temporal measurements (B1–B5, C1)\n\n_Skipped: `temporal_pairs.csv` not found (or empty). Run `measure_temporal.py`._\n")

    gonogo_md, printed = gonogo(timing_info, pixels_info, temporal_info)

    head = ["# Phase 0 — 2DGS visibility-caching feasibility report", "", f"Source directory: `{os.path.abspath(out_dir)}`", ""]
    head.append("Inputs: " + ", ".join(f"`{os.path.basename(p)}` ({'found, %d rows' % len(data[n]) if data[n] else 'missing'})" for n, p in paths.items()))
    if missing:
        head.append("")
        head.append("Missing inputs are reported as skipped sections and N/A criteria: " + ", ".join(f"`{m}`" for m in missing) + ".")
    head.append("")

    report = "\n".join(head + [gonogo_md] + [s + "\n" for s in sections])
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n".join(printed))
    print(f"\nWrote {report_path}")
    for png in ("plot_stages.png", "plot_iou.png", "plot_oracle.png", "plot_fallback.png"):
        p = os.path.join(out_dir, png)
        print(f"  {'wrote' if os.path.isfile(p) else 'skipped'} {p}")
    return report_path


# --------------------------------------------------------------------------- #
# --demo: synthetic CSVs with the exact WP-B / WP-C column names
# --------------------------------------------------------------------------- #
def write_demo_csvs(out_dir: str, seed: int = 0) -> None:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    jit = lambda s=0.05: 1.0 + rng.uniform(-s, s)  # noqa: E731
    N = 1_500_000
    views = [0, 20, 40, 60, 80]
    resolutions = [(1297, 840), (1920, 1244), (3840, 2488)]
    eps_list = [0.001, 0.005, 0.01, 0.05]
    stats = STAT_NAMES

    # premise_timing.csv
    with open(os.path.join(out_dir, "premise_timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "res_w", "res_h", "N", "n_isects", "t_projection_ms", "t_tiling_sort_ms", "t_sh_ms", "t_blend_ms", "t_total_ms", "blend_frac"])
        for v in views:
            for rw, rh in resolutions:
                npix = rw * rh
                tp, tt, ts, tb = 1.8 * jit(), (2.5 + 1.5e-6 * npix) * jit(), 1.2 * jit(), (1.0 + 3.0e-6 * npix) * jit(0.1)
                tot = tp + tt + ts + tb
                w.writerow([v, rw, rh, N, int(8e6 * (npix / 1.09e6) ** 0.7 * jit()), f"{tp:.3f}", f"{tt:.3f}", f"{ts:.3f}", f"{tb:.3f}", f"{tot:.3f}", f"{tb / tot:.4f}"])

    # premise_pixels.csv
    metrics = ["n_contrib", "n_contrib_eps", "n_eval", "tile_len", "acc_alpha", "ratio"]
    header = ["view", "res_w", "res_h", "N", "eps"] + [f"{m}_{s}" for m in metrics for s in stats] + ["psnr_trunc_eps"]
    eps_keep = {0.001: 0.8, 0.005: 0.55, 0.01: 0.45, 0.05: 0.25}
    with open(os.path.join(out_dir, "premise_pixels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for v in views:
            for rw, rh in resolutions:
                scale = (rw / 1297) ** 0.6  # more evaluated surfels at higher res, contributors ~flat
                for eps in eps_list:
                    c = {"mean": 28 * jit(), "p50": 24 * jit(), "p90": 60 * jit(), "p99": 110 * jit(), "max": 300 * jit()}
                    ce = {k: val * eps_keep[eps] for k, val in c.items()}
                    ev = {"mean": 180 * scale * jit(), "p50": 150 * scale * jit(), "p90": 380 * scale * jit(), "p99": 700 * scale * jit(), "max": 1400 * scale * jit()}
                    tl = {k: val * 1.25 for k, val in ev.items()}
                    aa = {"mean": 0.975, "p50": 0.995, "p90": 0.999, "p99": 0.999, "max": 1.0}
                    ratio = {k: ev[k] / max(ce[k], 1.0) for k in stats}
                    psnr = {0.001: 48, 0.005: 41, 0.01: 37, 0.05: 29}[eps] * jit(0.03)
                    row: list[Any] = [v, rw, rh, N, eps]
                    for d in (c, ce, ev, tl, aa, ratio):
                        row += [f"{d[s]:.4f}" for s in stats]
                    row.append(f"{psnr:.3f}")
                    w.writerow(row)

    # temporal_pairs.csv
    cols = ["view", "kind", "delta", "rot_deg", "trans_frac", "N", "eps", "K", "radius", "valid_frac", "iou_mean", "iou_p10", "iou_p50", "cap_bind_frac", "psnr_oracle", "lpips_oracle", "psnr_oracle_valid", "fallback_frac_tau0.01", "fallback_frac_tau0.05", "inversion_frac", "inv_depth_gap_p50", "psnr_stale_order", "union_frac", "quad_share", "psnr_stale_sh", "mean_candidates_per_pixel"]
    rot_deltas = [0.1, 0.25, 0.5, 1, 2, 4]
    trans_deltas = [0.25, 0.5, 1, 2, 4, 8]

    def synth_row(view: int, kind: str, delta: float, rot_deg: float, trans_frac: float, severity: float) -> list[Any]:
        s = max(severity, 1e-3)
        iou50 = min(0.995, 0.985 * math.exp(-s / 6.0) * jit(0.02))
        psnr = (44.0 - 4.0 * math.log2(1.0 + s / 0.25)) * jit(0.03)
        fb1 = min(0.5, 0.003 * s ** 0.9 * jit(0.2))
        return [view, kind, delta, f"{rot_deg:.4f}", f"{trans_frac:.5f}", N, 0.005, 32, 1, f"{0.97 - 0.01 * s:.4f}", f"{iou50 - 0.03:.4f}", f"{max(0.0, iou50 - 0.12):.4f}", f"{iou50:.4f}", f"{min(0.9, 0.05 + 0.03 * s):.4f}", f"{psnr:.3f}", f"{0.01 + 0.02 * s * jit(0.1):.4f}", f"{psnr + 0.7:.3f}", f"{fb1:.5f}", f"{fb1 * 0.5:.5f}", f"{0.01 + 0.012 * s:.4f}", f"{0.002 * (1 + s):.5f}", f"{psnr - 1.5 - 0.5 * s:.3f}", f"{0.12 + 0.005 * s:.4f}", f"{0.45 * jit(0.05):.4f}", f"{psnr - 0.8:.3f}", f"{45 * jit():.2f}"]

    with open(os.path.join(out_dir, "temporal_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for v in [0, 40, 80]:
            for kind in KINDS_ROT:
                for d in rot_deltas:
                    w.writerow(synth_row(v, kind, d, d * jit(0.01), 0.0, d))
            for kind in KINDS_TRANS:
                for d in trans_deltas:
                    w.writerow(synth_row(v, kind, d, 0.0, d / 100.0, d / 2.0))
        for step in range(1, 6):
            rd, tf = 0.4 * jit(0.3), 0.008 * jit(0.3)
            w.writerow(synth_row(0, "traj", step, rd, tf, rd + 100 * tf / 2.0))
    print(f"Demo CSVs written to {out_dir}")


DEFAULT_OUT = "examples/instrumentation/out"


def resolve_out(out: str | None) -> str:
    """Explicit --out wins; otherwise the PLAN default relative to the cwd if it
    exists, else ``<this script's dir>/out`` (identical when run from the repo
    root, and what ``cd examples; python instrumentation/report.py`` means)."""
    if out:
        return out
    if os.path.isdir(DEFAULT_OUT):
        return DEFAULT_OUT
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"directory holding the CSVs; report.md and PNGs are written here (default: {DEFAULT_OUT}, or instrumentation/out next to this script when run from examples/)")
    ap.add_argument("--demo", nargs="?", const="", default=None, metavar="DIR", help="write synthetic CSVs to DIR (default: a fresh temp dir) and run the report on them")
    args = ap.parse_args(argv)

    out = resolve_out(args.out)
    if args.demo is not None:
        out = args.demo or tempfile.mkdtemp(prefix="instrumentation_demo_")
        write_demo_csvs(out)
    build_report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
