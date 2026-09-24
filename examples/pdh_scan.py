"""Simulate [U-13C]glucose labeling at several PDH activities, add Gaussian
noise, and fit the PDH activity back from the noisy data.

Run:  python examples/pdh_scan.py [--mode compensated|uncompensated]

Writes to examples/output/{mode}/:
  * one CSV per plot panel, named from the panel's y-axis label, with columns
    pdh_factor, time_min, simulated, data, fit, fitted_pdh_factor, fitted_pdh_factor_se
    (``data`` is empty at times without a sampled point);
  * pdh_scan.png with all panels.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c13model import Parameters, add_gaussian_noise, fit_pdh, pdh_scan, simulate

FACTORS = [0.25, 0.5, 0.75, 1.0]  # PDH activity relative to baseline
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark = low -> high PDH
# (observable column, panel title, y-axis label)
PANELS = [
    ("Glu_C4_FE", "Glutamate C4", "Glu C4 fractional enrichment"),
    ("Glu_C3_FE", "Glutamate C3", "Glu C3 fractional enrichment"),
    ("Glu_C4_D45", "Glutamate C4 multiplet", "Glu C4 D45 fraction of C4 signal"),
]


def filename_from_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compensated", choices=["compensated", "uncompensated"])
    ap.add_argument("--noise-sd", type=float, default=0.02, help="absolute SD in fraction units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(__file__).parent / "output" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c, _, _ in PANELS]
    base = Parameters()
    t_fine = np.arange(0, 120.5, 0.5)
    sample_t = np.arange(0, 121, 5.0)

    model = pdh_scan(FACTORS, base, mode=args.mode, t_eval=t_fine)
    measured = add_gaussian_noise(model[model.time_min.isin(sample_t)], args.noise_sd, columns=cols, seed=args.seed)

    fits, fitted = {}, []
    for f in FACTORS:
        fit = fit_pdh(measured[measured.pdh_factor == f][["time_min", *cols]], cols, base, args.mode)
        fits[f] = fit
        curve = simulate(fit.params, t_eval=t_fine).observables()[cols].reset_index()
        curve.insert(0, "pdh_factor", f)
        fitted.append(curve)
        print(f"PDH x {f:g}: fitted {fit.factor:.3f} ± {fit.factor_se:.3f} (residual SD {fit.residual_sd:.4f})")
    fitted = pd.concat(fitted, ignore_index=True)

    keys = ["pdh_factor", "time_min"]
    for col, _, ylabel in PANELS:
        table = (
            model[keys + [col]].rename(columns={col: "simulated"})
            .merge(measured[keys + [col]].rename(columns={col: "data"}), on=keys, how="left")
            .merge(fitted[keys + [col]].rename(columns={col: "fit"}), on=keys)
        )
        table["fitted_pdh_factor"] = table.pdh_factor.map({f: r.factor for f, r in fits.items()})
        table["fitted_pdh_factor_se"] = table.pdh_factor.map({f: r.factor_se for f, r in fits.items()})
        table.to_csv(out / filename_from_label(ylabel), index=False, float_format="%.6g")
        print(f"Wrote {out / filename_from_label(ylabel)}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14, 4.2), constrained_layout=True)
    for ax, (col, title, ylabel) in zip(axes, PANELS):
        for f, color in zip(FACTORS, COLORS):
            m = model[model.pdh_factor == f]
            s = measured[measured.pdh_factor == f]
            ft = fitted[fitted.pdh_factor == f]
            ax.plot(m.time_min, m[col], color=color, lw=2, label=f"PDH × {f:g}")
            ax.plot(ft.time_min, ft[col], color=color, lw=1.5, ls="--")
            ax.scatter(s.time_min, s[col], s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.tick_params(colors="#55544f")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#55544f", lw=2), plt.Line2D([], [], color="#55544f", lw=1.5, ls="--"),
                plt.Line2D([], [], color="#55544f", marker="o", ls="", ms=4)]
    labels += ["simulated", "fit", "data (+ noise)"]
    order = [3, 2, 1, 0, 4, 5, 6]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="outside right upper")
    fig.suptitle(
        f"[U-13C]glucose, {args.mode} PDH scan (Gaussian noise SD {args.noise_sd:g}; PDH fitted jointly to all panels)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.savefig(out / "pdh_scan.png", dpi=150)
    print(f"Wrote {out / 'pdh_scan.png'}")


if __name__ == "__main__":
    main()
