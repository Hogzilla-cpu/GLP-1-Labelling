"""Plot the per-panel CSV files written by examples/scan.py.

Each CSV is plotted to a PNG of the same name next to it; the y-axis label is
recovered from the file name.  Also writes overview.png per folder with all
panels (glutamate top row, glutamine bottom row).

  python examples/plot_csv.py docs/glucose_pdh_compensated_both
  python examples/plot_csv.py docs/*/                 # every scan folder
  python examples/plot_csv.py path/to/Glu_C4_fractional_enrichment.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark = low -> high activity
MUTED, GRID = "#55544f", "#e5e4e0"


def label_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ")


def factor_column(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if c.endswith("_factor") and not c.startswith("fitted_"))


def plot_csv(path: Path, ax: plt.Axes) -> None:
    df = pd.read_csv(path)
    fcol = factor_column(df)
    enzyme = fcol.split("_")[0].upper()
    ylabel = label_from_filename(path)

    if "time_min" in df.columns:  # time course: one series per activity level
        levels = sorted(df[fcol].unique())
        colors = COLORS[-len(levels):] if len(levels) <= len(COLORS) else plt.cm.Blues_r(range(len(levels)))
        for f, color in zip(levels, colors):
            d = df[df[fcol] == f]
            ax.plot(d.time_min, d.simulated, color=color, lw=2, label=f"{enzyme} × {f:g}")
            if d.fit.notna().any():
                ax.plot(d.time_min, d.fit, color=color, lw=1.5, ls="--")
            ax.scatter(d.time_min, d.data, s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xlabel("Time (min)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=8)
    else:  # labeling vs. activity
        ax.plot(df[fcol], df.simulated, color=MUTED, lw=2, label="simulated")
        pts = df[df.data.notna()].reset_index(drop=True)
        colors = COLORS[-len(pts):]
        for (_, r), color in zip(pts.iterrows(), colors):
            ax.scatter(r[fcol], r.data, s=30, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(r[f"fitted_{fcol}"], r.fit, s=40, marker="D", facecolor="none", edgecolor=color,
                       linewidth=1.5, zorder=4)
        ax.set_xlabel(f"{enzyme} activity (× baseline)")
        ax.legend([plt.Line2D([], [], color=MUTED, lw=2),
                   plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=5),
                   plt.Line2D([], [], color=MUTED, marker="D", ls="", ms=5, mfc="none")],
                  ["simulated", "data (+ noise)", f"fit (at fitted {enzyme})"], frameon=False, fontsize=8)
        lo, hi = ax.get_ylim()
        if hi - lo < 0.2:  # don't magnify negligible effects
            mid = (hi + lo) / 2
            ax.set_ylim(mid - 0.1, mid + 0.1)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="CSV files or folders of CSVs")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 10})

    for path in args.paths:
        csvs = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for csv in csvs:
            fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
            plot_csv(csv, ax)
            fig.savefig(csv.with_suffix(".png"), dpi=150)
            plt.close(fig)
            print(f"Wrote {csv.with_suffix('.png')}")

        if path.is_dir() and csvs:
            rows = [[c for c in csvs if c.name.startswith(m)] for m in ("Glu", "Gln")]
            rows = [sorted(r, key=lambda c: ("_at_" in c.name, "_C3_" in c.name)) for r in rows]
            fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True, squeeze=False)
            for r, row in enumerate(rows):
                for c, csv in enumerate(row[:3]):
                    plot_csv(csv, axes[r][c])
            fig.suptitle(path.name.replace("_", " "), x=0.01, ha="left", fontsize=12)
            fig.savefig(path / "overview.png", dpi=120)
            plt.close(fig)
            print(f"Wrote {path / 'overview.png'}")


if __name__ == "__main__":
    main()
