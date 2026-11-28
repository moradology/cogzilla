#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=1.26"]
# ///
"""Generate 3 publication-grade benchmark charts for docs/benchmarks.md and docs/gpu-benchmarks.md.

Run with zero global installs:
    uv run benches/plot_benchmarks.py
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

BG = "#0d1117"
PANEL = "#131922"
SPINE = "#30363d"
GRID = "#21262d"
TEXT = "#e6edf3"
TITLE = "#f0f6fc"
MUTED = "#8b949e"

# Colorblind-safe, high-contrast dark-mode palette
C_BLUE   = "#58a6ff"   # cogzilla — Threadripper 3960X / RTX 3090 Ti
C_GREEN  = "#3fb950"   # cogzilla — Apple M5 (CPU / Metal)
C_AMBER  = "#f0883e"   # TiTiler 2.4 — Threadripper 3960X / CPU baseline
C_RED    = "#f85149"   # TiTiler 2.4 — Apple M5 / AIMD OFF
C_PURPLE = "#bc8cff"   # Apple M5 1-core CPU baseline
C_GRAY   = "#6e7681"   # Neutral baseline

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.9,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TITLE,
    "axes.titlepad": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.facecolor": BG,
    "legend.edgecolor": SPINE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})


def style_ax(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, which="major", alpha=0.45)
    ax.set_axisbelow(True)


def render_titiler_scaling() -> None:
    """Chart 1 (2x2): Apples-to-apples cogzilla vs TiTiler 2.4 throughput & latency scaling
    on 256x256 RGB Bilinear PNG (top) and 512x512 RGB Lanczos3 JPEG (bottom).
    All numbers are 3-run medians at commit 3787b88 with encoded-tile cache disabled."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.2, 9.6), dpi=220)
    fig.suptitle(
        "cogzilla vs. TiTiler 2.4 — Throughput & Latency Scaling",
        fontsize=15, fontweight="bold", y=0.975, color=TITLE,
    )
    fig.text(
        0.5, 0.942,
        "Same 2048×2048 3-band F32 COG, 726-tile Web Mercator pool, encoded-tile cache OFF, 3-run medians (3960X: 3787b88, M5: f0aa40e)",
        ha="center", fontsize=9.8, color=MUTED,
    )

    # Shared legend cleanly placed above the 2x2 grid
    handles = [
        Line2D([0], [0], color=C_BLUE,  marker="o", lw=2.5, ms=6.5, label="cogzilla — Threadripper 3960X 24C/48T (185–428 MiB RSS)"),
        Line2D([0], [0], color=C_GREEN, marker="s", lw=2.5, ms=6.5, label="cogzilla — Apple M5 10C (206–343 MiB RSS)"),
        Line2D([0], [0], color=C_AMBER, marker="^", lw=2.2, ms=6.5, ls="--", label="TiTiler 2.4 — Threadripper 3960X 1–48w (154–7,021 MiB RSS)"),
        Line2D([0], [0], color=C_RED,   marker="D", lw=2.2, ms=6.0, ls="--", label="TiTiler 2.4 — Apple M5 1–10w (174–1,543 MiB RSS)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.922),
        ncol=2, frameon=True, fontsize=8.8, handlelength=2.4, columnspacing=1.8,
    )

    # --- Top row: HTTP 256x256 RGB Bilinear PNG (1..48 cores/workers) ---
    w_tr = np.array([1, 2, 4, 8, 16, 24, 32, 48])
    cz_tr_rps = np.array([547.4, 1044.9, 1939.2, 3421.4, 5551.8, 6985.4, 7784.4, 7622.1])
    cz_tr_p50 = np.array([1.48,  1.60,   1.80,   2.11,   2.67,   3.24,   3.87,   5.31])

    w_tt_tr = np.array([1, 2, 4, 8, 16, 24, 48])
    tt_tr_rps = np.array([45.9,  69.9,  206.5, 267.4, 407.4, 544.5, 951.8])
    tt_tr_p50 = np.array([19.31, 27.23, 17.77, 26.80, 36.29, 40.35, 36.18])

    w_m5 = np.array([1, 2, 4, 8, 10])
    cz_m5_rps = np.array([1610.0, 1818.5, 4671.9, 7514.8, 8427.3])
    cz_m5_p50 = np.array([0.55,   1.06,   0.81,   0.98,   1.07])
    tt_m5_rps = np.array([118.8,  236.4,  245.5,  373.0,  418.6])
    tt_m5_p50 = np.array([7.69,   18.98,  29.23,  34.85,  50.44])

    # Panel A: 256^2 Throughput
    ax1.plot(w_tr,    cz_tr_rps, "-o",  color=C_BLUE,  lw=2.5, ms=6)
    ax1.plot(w_m5,    cz_m5_rps, "-s",  color=C_GREEN, lw=2.5, ms=6)
    ax1.plot(w_tt_tr, tt_tr_rps, "--^", color=C_AMBER, lw=2.1, ms=6)
    ax1.plot(w_m5,    tt_m5_rps, "--D", color=C_RED,   lw=2.1, ms=5.5)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([1, 2, 4, 8, 10, 16, 24, 32, 48])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlim(0.85, 76)
    ax1.set_ylim(30, 16000)
    ax1.set_xlabel("Concurrency / TiTiler Worker Processes")
    ax1.set_ylabel("Throughput (req/s, higher is better)")
    ax1.set_title("A. 256×256 RGB Bilinear PNG — Throughput")
    style_ax(ax1, "both")
    ax1.text(48 * 1.08, 7622, "7,622 r/s\n(8.0×)", color=C_BLUE,  fontsize=8.3, fontweight="bold", va="center")
    ax1.text(10 * 1.12, 8427, "8,427 r/s (20.1×)",  color=C_GREEN, fontsize=8.3, fontweight="bold", va="bottom")
    ax1.text(48 * 1.08, 952,  "952 r/s",            color=C_AMBER, fontsize=8.2, va="center")
    ax1.text(10 * 1.12, 419,  "419 r/s",            color=C_RED,   fontsize=8.2, va="center")

    # Panel B: 256^2 Median Latency
    ax2.plot(w_tt_tr, tt_tr_p50, "--^", color=C_AMBER, lw=2.1, ms=6)
    ax2.plot(w_m5,    tt_m5_p50, "--D", color=C_RED,   lw=2.1, ms=5.5)
    ax2.plot(w_tr,    cz_tr_p50, "-o",  color=C_BLUE,  lw=2.5, ms=6)
    ax2.plot(w_m5,    cz_m5_p50, "-s",  color=C_GREEN, lw=2.5, ms=6)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks([1, 2, 4, 8, 10, 16, 24, 32, 48])
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlim(0.85, 76)
    ax2.set_ylim(0.35, 95)
    ax2.set_xlabel("Concurrency / TiTiler Worker Processes")
    ax2.set_ylabel("Median Latency p50 (ms, lower is better)")
    ax2.set_title("B. 256×256 RGB Bilinear PNG — Median Latency")
    style_ax(ax2, "both")
    ax2.text(10 * 1.12, 50.4, "50.4 ms",            color=C_RED,   fontsize=8.2, va="center")
    ax2.text(48 * 1.08, 36.2, "36.2 ms",            color=C_AMBER, fontsize=8.2, va="center")
    ax2.text(48 * 1.08, 5.31, "5.3 ms\n(6.8×)",     color=C_BLUE,  fontsize=8.3, fontweight="bold", va="center")
    ax2.text(10 * 1.12, 1.07, "1.1 ms (47× lower)", color=C_GREEN, fontsize=8.3, fontweight="bold", va="center")

    # --- Bottom row: 512x512 RGB Lanczos3 JPEG (c = 1, 4, 16, 32 — single harness, zero mixing) ---
    c4 = np.array([1, 4, 16, 32])
    tr_512_rps = np.array([349, 1381, 5299, 7908])
    m5_512_rps = np.array([1146, 3387, 5432, 5502])
    tt_tr_512_rps = np.array([12.9, 44.8, 155.6, 239.9])
    tt_m5_512_rps = np.array([26.2, 72.7, 99.5,  103.8])

    tr_512_p50 = np.array([2.6, 2.6, 2.7, 4.2])
    tr_512_p99 = np.array([8.1, 8.2, 8.7, 12.8])
    m5_512_p50 = np.array([0.7, 1.0, 2.4, 5.4])
    m5_512_p99 = np.array([3.1, 4.2, 12.2, 15.1])
    tt_tr_512_p50 = np.array([89.0, 91.6, 101.1, 129.6])
    tt_tr_512_p99 = np.array([126.3, 121.8, 137.1, 194.3])
    tt_m5_512_p50 = np.array([42.1, 59.0, 158.2, 309.7])
    tt_m5_512_p99 = np.array([53.2, 72.1, 245.0, 415.3])

    # Panel C: 512^2 Lanczos3 Throughput
    ax3.plot(c4, tr_512_rps,    "-o",  color=C_BLUE,  lw=2.5, ms=6.5)
    ax3.plot(c4, m5_512_rps,    "-s",  color=C_GREEN, lw=2.5, ms=6.5)
    ax3.plot(c4, tt_tr_512_rps, "--^", color=C_AMBER, lw=2.1, ms=6.5)
    ax3.plot(c4, tt_m5_512_rps, "--D", color=C_RED,   lw=2.1, ms=6.0)
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_xticks([1, 4, 16, 32])
    ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlim(0.85, 58)
    ax3.set_ylim(8, 18000)
    ax3.set_xlabel("Concurrent Clients (1 = single core)")
    ax3.set_ylabel("Throughput (req/s, higher is better)")
    ax3.set_title("C. 512×512 RGB Lanczos3 JPEG — Throughput")
    style_ax(ax3, "both")
    ax3.text(32 * 1.08, 7908, "7,908 r/s\n(33×)", color=C_BLUE,  fontsize=8.3, fontweight="bold", va="bottom")
    ax3.text(32 * 1.08, 5502, "5,502 r/s\n(53×)", color=C_GREEN, fontsize=8.3, fontweight="bold", va="top")
    ax3.text(32 * 1.08, 240,  "240 r/s",          color=C_AMBER, fontsize=8.2, va="center")
    ax3.text(32 * 1.08, 104,  "104 r/s",          color=C_RED,   fontsize=8.2, va="center")

    # Panel D: 512^2 Lanczos3 Latency (p50 solid + shaded p50->p99 band)
    for x, p50, p99, col, sty, mk in [
        (c4, tt_m5_512_p50, tt_m5_512_p99, C_RED,   "--", "D"),
        (c4, tt_tr_512_p50, tt_tr_512_p99, C_AMBER, "--", "^"),
        (c4, tr_512_p50,    tr_512_p99,    C_BLUE,  "-",  "o"),
        (c4, m5_512_p50,    m5_512_p99,    C_GREEN, "-",  "s"),
    ]:
        ax4.fill_between(x, p50, p99, color=col, alpha=0.14)
        ax4.plot(x, p50, linestyle=sty, marker=mk, color=col, lw=2.4 if sty == "-" else 2.1, ms=6)
    ax4.set_xscale("log", base=2)
    ax4.set_yscale("log")
    ax4.set_xticks([1, 4, 16, 32])
    ax4.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax4.set_xlim(0.85, 76)
    ax4.set_ylim(0.5, 700)
    ax4.set_xlabel("Concurrent Clients (1 = single core)")
    ax4.set_ylabel("Latency (ms, lower is better)")
    ax4.set_title("D. 512×512 RGB Lanczos3 JPEG — Latency (p50 Line, p99 Shaded)")
    style_ax(ax4, "both")
    ax4.text(32 * 1.08, 310, "310 ms",            color=C_RED,   fontsize=8.2, va="center")
    ax4.text(32 * 1.08, 130, "130 ms",            color=C_AMBER, fontsize=8.2, va="center")
    ax4.text(32 * 1.08, 6.2, "5.4 ms\n(p99 15ms)", color=C_GREEN, fontsize=8.1, fontweight="bold", va="bottom")
    ax4.text(32 * 1.08, 3.6, "4.2 ms\n(p99 13ms)", color=C_BLUE,  fontsize=8.1, fontweight="bold", va="top")

    fig.text(
        0.5, 0.012,
        "TiTiler 2.4 uses uvicorn + uvloop (10w on M5 = 1.5 GiB RSS; 48w on 3960X = 7.0–7.7 GiB RSS). "
        "cogzilla uses a single process (168–428 MiB RSS — up to 16.4× less RAM).",
        ha="center", fontsize=8.6, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.032, 1, 0.855])
    fig.savefig(ASSETS / "bench-concurrency-overload.png", dpi=220)
    plt.close(fig)


def render_gpu_and_overload() -> None:
    """Chart 2 (2x2): GPU acceleration (horizontal dumbbell for heavy endpoints + p50 soak
    comparison) and live HTTP AIMD overload protection (goodput preserved + tail latency)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.4, 9.8), dpi=220)
    fig.suptitle(
        "GPU Acceleration (wgpu Metal & Vulkan) & Live HTTP AIMD Overload Protection",
        fontsize=15, fontweight="bold", y=0.978, color=TITLE,
    )
    fig.text(
        0.5, 0.946,
        "Heavy single-request GPU endpoints (1-core CPU vs. GPU), sustained soak p50 latency, and 128–256c HTTP overload (3-run medians at 3787b88)",
        ha="center", fontsize=9.6, color=MUTED,
    )

    # --- Panel A (Top-Left): Horizontal Dumbbell — Heavy Single-Request Endpoints ---
    # Sorted by RTX 3090 Ti latency; excludes pre-PR#571 4096^2 RGB Lanczos3 baseline.
    endpoints = [
        "4096² UTM→WGS84\nGrid2D Stats",
        "2048² 4-Band F32\n/cog/statistics",
        "4096² Polygon\nMasked Stats",
        "2048²×12 Median\nMosaic → JPEG",
        "2048² Exact\n4-Band Zonal",
    ]
    rtx_ms    = np.array([1.33, 1.37, 4.36, 31.12, 36.29])
    tr_cpu_ms = np.array([934.3, 581.6, 372.8, 239.6, 341.1])
    m5g_ms    = np.array([8.93, 5.28, 7.64, 33.12, 17.66])
    m5_cpu_ms = np.array([379.5, 251.9, 180.3, 145.7, 102.4])

    y = np.arange(len(endpoints))
    dy = 0.17

    for i in range(len(endpoints)):
        # Top track: 3960X CPU -> RTX 3090 Ti
        ax1.plot([rtx_ms[i], tr_cpu_ms[i]], [y[i] - dy, y[i] - dy], color=C_BLUE, lw=2.2, alpha=0.75)
        # Bottom track: M5 CPU -> M5 Metal
        ax1.plot([m5g_ms[i], m5_cpu_ms[i]], [y[i] + dy, y[i] + dy], color=C_GREEN, lw=2.2, alpha=0.75)

    ax1.scatter(tr_cpu_ms, y - dy, color=C_AMBER,  s=52, zorder=4, marker="^", label="3960X CPU (1 core)")
    ax1.scatter(rtx_ms,    y - dy, color=C_BLUE,   s=58, zorder=5, marker="o", label="RTX 3090 Ti (Vulkan)")
    ax1.scatter(m5_cpu_ms, y + dy, color=C_PURPLE, s=48, zorder=4, marker="D", label="Apple M5 CPU (1 core)")
    ax1.scatter(m5g_ms,    y + dy, color=C_GREEN,  s=56, zorder=5, marker="s", label="Apple M5 GPU (Metal)")

    for i in range(len(endpoints)):
        sp_rtx = tr_cpu_ms[i] / rtx_ms[i]
        sp_m5  = m5_cpu_ms[i] / m5g_ms[i]
        mid_rtx = np.sqrt(rtx_ms[i] * tr_cpu_ms[i])
        mid_m5  = np.sqrt(m5g_ms[i] * m5_cpu_ms[i])
        lbl_rtx = f"{sp_rtx:.0f}×" if sp_rtx >= 10 else f"{sp_rtx:.1f}×"
        lbl_m5  = f"{sp_m5:.0f}×" if sp_m5 >= 10 else f"{sp_m5:.1f}×"
        ax1.text(mid_rtx, y[i] - dy - 0.08, lbl_rtx, ha="center", va="bottom",
                 fontsize=7.8, color="#79c0ff", fontweight="bold")
        ax1.text(mid_m5,  y[i] + dy + 0.09, lbl_m5,  ha="center", va="top",
                 fontsize=7.6, color="#7ee787", fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xlim(0.7, 2200)
    ax1.set_ylim(4.65, -0.55)
    ax1.set_yticks(y)
    ax1.set_yticklabels(endpoints, fontsize=8.5)
    ax1.set_xlabel("Single-Request Latency (ms, log₁₀, lower is better)")
    ax1.set_title("A. Heavy Endpoints: 1-Core CPU → GPU Speedup")
    style_ax(ax1, "x")
    ax1.legend(loc="lower left", fontsize=7.6, ncol=1, framealpha=0.92)

    # --- Panel B (Top-Right): Sustained Soak p50 Latency ONLY (consistent metric!) ---
    soak_cases = [
        "Apple M5\nHeavy 2048²\n(under 12c load)",
        "3960X + 3090 Ti\nHeavy 2048²\n(under 12c load)",
        "3960X + 3090 Ti\nMixed Flood\n(32c, 0ms think)",
        "3960X + 3090 Ti\nPan & Zoom\n(16c mixed)",
    ]
    soak_cpu_p50 = [120.2, 237.7, 3.0, 2.8]
    soak_gpu_p50 = [34.6,  46.4,  2.5, 2.3]
    soak_gpu_rps = ["56 vs 11 r/s", "45 vs 7 r/s", "11,197 vs 8,400", "2,560 vs 2,215"]

    xb = np.arange(len(soak_cases))
    wb = 0.34
    ax2.bar(xb - wb / 2, soak_cpu_p50, width=wb, color=C_AMBER, alpha=0.88, label="CPU-Only (all cores) — p50")
    ax2.bar(xb + wb / 2, soak_gpu_p50, width=wb, color=C_BLUE,  alpha=0.92, label="GPU auto (coalesced) — p50")
    ax2.set_yscale("log")
    ax2.set_ylim(1, 700)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(soak_cases, fontsize=8.3)
    ax2.set_ylabel("Median Latency p50 (ms, log₁₀, lower is better)")
    ax2.set_title("B. Sustained Map-Server Soak — Median Latency (p50)")
    style_ax(ax2, "y")

    for i in range(len(soak_cases)):
        ax2.text(xb[i] - wb / 2, soak_cpu_p50[i] * 1.12, f"{soak_cpu_p50[i]:.1f}ms",
                 ha="center", va="bottom", fontsize=8.0, color="#ffa657")
        gain = soak_cpu_p50[i] / soak_gpu_p50[i]
        ax2.text(xb[i] + wb / 2, soak_gpu_p50[i] * 1.12, f"{soak_gpu_p50[i]:.1f}ms\n({gain:.1f}×)",
                 ha="center", va="bottom", fontsize=8.0, color="#79c0ff", fontweight="bold")
        ax2.text(xb[i], 1.25, soak_gpu_rps[i], ha="center", va="bottom",
                 fontsize=7.4, color=MUTED, bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=SPINE, alpha=0.9))
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    # --- Panel C (Bottom-Left): Live HTTP AIMD Overload — 200 OK Goodput Preserved ---
    hosts = [
        "Apple M5\nCPU (10C)\n128 clients",
        "Apple M5\nGPU auto\n256 clients",
        "Threadripper\n3960X (48T)\n128 clients",
        "RTX 3090 Ti\n+ 3960X\n256 clients",
    ]
    goodput_off = [3657.4, 3711.8, 7934.2, 8034.0]
    goodput_on  = [3619.2, 3508.0, 7834.3, 7675.1]

    xh = np.arange(len(hosts))
    ax3.bar(xh - wb / 2, goodput_off, width=wb, color=C_GRAY,  alpha=0.85, label="AIMD OFF — 200 OK Goodput")
    ax3.bar(xh + wb / 2, goodput_on,  width=wb, color=C_GREEN, alpha=0.92, label="AIMD ON — 200 OK Goodput")
    ax3.set_ylim(0, 10200)
    ax3.set_xticks(xh)
    ax3.set_xticklabels(hosts, fontsize=8.5)
    ax3.set_ylabel("Goodput (200 OK req/s, higher is better)")
    ax3.set_title("C. Live HTTP Overload — Goodput Preserved (95–99%)")
    style_ax(ax3, "y")

    for i in range(len(hosts)):
        pct = 100.0 * goodput_on[i] / goodput_off[i]
        ax3.text(xh[i] - wb / 2, goodput_off[i] + 160, f"{goodput_off[i]:,.0f}",
                 ha="center", va="bottom", fontsize=8.0, color="#c9d1d9")
        ax3.text(xh[i] + wb / 2, goodput_on[i] + 160, f"{goodput_on[i]:,.0f}\n({pct:.0f}%)",
                 ha="center", va="bottom", fontsize=8.0, color="#7ee787", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    # --- Panel D (Bottom-Right): Live HTTP AIMD Overload — Latency (zero box over bars!) ---
    p99_off = [290.3, 448.9, 33.4, 69.9]
    p99_on  = [97.5,  181.0, 33.8, 56.8]
    p50_on  = [5.5,   10.5,  17.4, 26.9]

    w3 = 0.26
    ax4.bar(xh - w3, p99_off, width=w3, color=C_RED,   alpha=0.85, label="AIMD OFF — 200 OK p99")
    ax4.bar(xh,      p99_on,  width=w3, color=C_BLUE,  alpha=0.92, label="AIMD ON — 200 OK p99")
    ax4.bar(xh + w3, p50_on,  width=w3, color=C_GREEN, alpha=0.92, label="AIMD ON — 200 OK p50")
    ax4.axhline(100.0, color=MUTED, ls=":", lw=1.1, alpha=0.7)
    ax4.text(2.42, 110, "100 ms target", ha="center", va="bottom", fontsize=7.6, color=MUTED)

    ax4.set_yscale("log")
    ax4.set_ylim(2, 1300)
    ax4.set_xticks(xh)
    ax4.set_xticklabels(hosts, fontsize=8.5)
    ax4.set_ylabel("Latency (ms, log₁₀, lower is better)")
    ax4.set_title("D. Live HTTP Overload — Tail (p99) & Median (p50) Latency")
    style_ax(ax4, "y")

    for i in range(len(hosts)):
        gain = p99_off[i] / p99_on[i]
        lbl = f"{p99_on[i]:.0f}ms\n({gain:.1f}×)" if gain >= 1.15 else f"{p99_on[i]:.0f}ms"
        ax4.text(xh[i], p99_on[i] * 1.14, lbl,
                 ha="center", va="bottom", fontsize=8.0, color="#79c0ff", fontweight="bold")
        ax4.text(xh[i] - w3, p99_off[i] * 1.09, f"{p99_off[i]:.0f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#ffa198")
        ax4.text(xh[i] + w3, p50_on[i] * 1.10, f"{p50_on[i]:.1f}ms",
                 ha="center", va="bottom", fontsize=7.8, color="#7ee787")
    ax4.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    fig.text(
        0.5, 0.012,
        "AIMD sheds excess overload with HTTP 429 in 0.3–3.4 ms while keeping /healthz p99 at 1.9–40.1 ms. "
        "All soak & AIMD rows are 3-run medians on 512² Lanczos3 / mixed map traffic (tile cache OFF).",
        ha="center", fontsize=8.6, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.032, 1, 0.928])
    fig.savefig(ASSETS / "bench-workload-contention.png", dpi=220)
    plt.close(fig)


def render_s3_cache_scaling() -> None:
    """Chart 3 (1x2): Real S3 COGs measured cold vs warm CPU cost (left) and controlled
    HTTP Range origin verification of GDAL/TiTiler caching vs cogzilla (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8), dpi=220, gridspec_kw={"width_ratios": [1.22, 1.0]})
    fig.suptitle(
        "Cloud Storage Caching — Real S3 COGs & Verified GDAL Range-Request Behavior",
        fontsize=14.5, fontweight="bold", y=0.975, color=TITLE,
    )
    fig.text(
        0.5, 0.925,
        "Apple M5 (10 cores) — TiTiler tuned with GDAL_CACHEMAX=200, CPL_VSIL_CURL_CACHE_SIZE=200MB, VSI_CACHE=TRUE",
        ha="center", fontsize=9.4, color=MUTED,
    )

    # --- Panel A (Left): Measured Real S3 COGs Cold Miss vs Warm Cache (Horizontal Grouped Bars) ---
    workloads = [
        "Equal Mix\n(5 S3 COGs)",
        "3-Asset STAC\nRGB Composite",
        "USGS 3DEP\nLZW DEM",
        "Sentinel-2\nB08 Viridis",
        "Sentinel-2\nTCI JPEG",
    ]
    tt_miss = [56.0, 116.0, 66.0, 36.3, 30.8]
    tt_hot  = [55.3, 120.0, 70.0, 32.8, 26.5]
    cz_miss = [13.2, 25.4,  13.7, 8.6,  14.4]
    cz_hot  = [1.7,  2.8,   1.1,  1.6,  2.1]

    y = np.arange(len(workloads))
    h = 0.20
    ax1.barh(y - 1.5 * h, tt_miss, height=h, color=C_RED,   alpha=0.55, label="TiTiler — Cold Miss (0% cache)")
    ax1.barh(y - 0.5 * h, tt_hot,  height=h, color=C_RED,   alpha=0.92, label="TiTiler — Repeated Tiles (hot)")
    ax1.barh(y + 0.5 * h, cz_miss, height=h, color=C_BLUE,  alpha=0.60, label="cogzilla — Cold Miss (0% cache)")
    ax1.barh(y + 1.5 * h, cz_hot,  height=h, color=C_GREEN, alpha=0.95, label="cogzilla — Cached Blocks (hot)")

    ax1.set_xscale("log")
    ax1.set_xlim(0.7, 320)
    ax1.set_yticks(y)
    ax1.set_yticklabels(workloads, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("CPU-Seconds per 1,000 Tiles (log₁₀, lower is better)")
    ax1.set_title("A. Real S3 COGs: Measured Cold Miss vs. Warm Cache")
    style_ax(ax1, "x")

    for i in range(len(workloads)):
        ratio_hot = tt_hot[i] / cz_hot[i]
        ax1.text(tt_hot[i] * 1.09, y[i] - 0.5 * h, f"{tt_hot[i]:.1f}", va="center", fontsize=7.8, color="#ffa198")
        ax1.text(cz_miss[i] * 1.09, y[i] + 0.5 * h, f"{cz_miss[i]:.1f}", va="center", fontsize=7.8, color="#79c0ff")
        ax1.text(cz_hot[i] * 1.12, y[i] + 1.5 * h, f"{cz_hot[i]:.1f} ({ratio_hot:.0f}× less)",
                 va="center", fontsize=8.0, color="#7ee787", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    # --- Panel B (Right): Controlled HTTP Range Origin Verification (3.12 MiB DEFLATE COG, c=8) ---
    cfgs = [
        "TiTiler 2.4\nMERGE_RANGES=YES\n(docs default)",
        "TiTiler 2.4\nMERGE_RANGES=NO\n(/vsicurl/ cache)",
        "cogzilla CPU\n--cache-mb 512\n(tile-cache=0)",
    ]
    hot_cpu  = [25.0, 25.9, 0.7]
    hot_reqs = [4.19, 4.17, 0.007]

    xr = np.arange(len(cfgs))
    w = 0.34
    ax2.bar(xr - w / 2, hot_cpu,  width=w, color=C_BLUE,  alpha=0.92, label="Hot CPU-s / 1,000 tiles")
    ax2.bar(xr + w / 2, hot_reqs, width=w, color=C_AMBER, alpha=0.92, label="Origin HTTP Range Reqs / tile (hot)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.004, 110)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(cfgs, fontsize=8.4)
    ax2.set_ylabel("Value (log₁₀, lower is better)")
    ax2.set_title("B. Counting Origin Proof (3.12 MiB DEFLATE COG)")
    style_ax(ax2, "y")

    for i in range(len(cfgs)):
        cpu_lbl = f"{hot_cpu[i]:.1f}\n(36× less)" if i == 2 else f"{hot_cpu[i]:.1f}"
        req_lbl = f"{hot_reqs[i]:.3f}" if i == 2 else f"{hot_reqs[i]:.2f}"
        ax2.text(xr[i] - w / 2, hot_cpu[i] * 1.15, cpu_lbl, ha="center", va="bottom",
                 fontsize=8.2, color="#79c0ff", fontweight="bold")
        ax2.text(xr[i] + w / 2, hot_reqs[i] * 1.18, f"{req_lbl}\nreqs/tile", ha="center", va="bottom",
                 fontsize=8.0, color="#ffa657", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.text(
        0.5, 0.015,
        "GDAL caches COG headers (miss→hot drops 36.9→25.0 CPU-s/1k on the 3.12 MiB COG), but rio-tiler closes rasterio.open() per request "
        "(flushing GDAL_CACHEMAX & per-handle VSI_CACHE) and GTiff multi-range reads bypass /vsicurl/.",
        ha="center", fontsize=8.3, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.895])
    fig.savefig(ASSETS / "bench-s3-cache-scaling.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_titiler_scaling()
    render_gpu_and_overload()
    render_s3_cache_scaling()
    print(f"Wrote 3 publication-grade benchmark charts to {ASSETS}")
