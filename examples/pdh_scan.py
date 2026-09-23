"""Simulate [U-13C]glucose labeling at several PDH activities, with noise.

Run:  python examples/pdh_scan.py [--mode compensated|uncompensated]
Writes examples/output/pdh_scan_{mode}.csv (noisy "measurements") and a PNG.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from c13model import Parameters, add_gaussian_noise, pdh_scan

FACTORS = [0.25, 0.5, 0.75, 1.0]  # PDH activity relative to baseline
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark = low -> high PDH
PANELS = [
    ("Glu_C4_FE", "Glutamate C4 enrichment"),
    ("Glu_C3_FE", "Glutamate C3 enrichment"),
    ("Glu_C4_D45", "Glutamate C4 D45 fraction"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compensated", choices=["compensated", "uncompensated"])
    ap.add_argument("--noise-sd", type=float, default=0.02, help="absolute SD in FE units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    model = pdh_scan(FACTORS, Parameters(), mode=args.mode, t_end=120)
    sample_t = np.arange(0, 121, 5.0)
    measured = add_gaussian_noise(
        model[model.time_min.isin(sample_t)], args.noise_sd,
        columns=[c for c, _ in PANELS], seed=args.seed,
    )
    measured.to_csv(out / f"pdh_scan_{args.mode}.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14, 4), constrained_layout=True)
    for ax, (col, title) in zip(axes, PANELS):
        for f, color in zip(FACTORS, COLORS):
            m = model[model.pdh_factor == f]
            s = measured[measured.pdh_factor == f]
            ax.plot(m.time_min, m[col], color=color, lw=2, label=f"PDH × {f:g}")
            ax.scatter(s.time_min, s[col], s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Time (min)")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.tick_params(colors="#55544f")
    axes[0].set_ylabel("Fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], frameon=False, loc="outside right upper")
    fig.suptitle(
        f"[U-13C]glucose, {args.mode} PDH scan (lines: model; points: + Gaussian noise, SD {args.noise_sd:g})",
        x=0.01, ha="left", fontsize=12,
    )
    fig.savefig(out / f"pdh_scan_{args.mode}.png", dpi=150)
    print(f"Wrote {out}/pdh_scan_{args.mode}.csv and .png")


if __name__ == "__main__":
    main()
