"""Simulate 13C labeling at several PDH or LDH activities, add Gaussian noise,
and fit the activity back from the noisy data.

Run, for example:
  python examples/scan.py                                   # [U-13C]glucose, PDH
  python examples/scan.py --enzyme ldh                      # [U-13C]glucose, LDH
  python examples/scan.py --substrate acetate               # [2-13C]acetate, PDH
  python examples/scan.py --substrate acetate --acetate-tracer 1,2-13C

Writes to examples/output/{substrate}_{enzyme}_{mode}/:
  * one CSV per plot panel, named from the panel's y-axis label, with columns
    {enzyme}_factor, time_min, simulated, data, fit, fitted_{enzyme}_factor,
    fitted_{enzyme}_factor_se  (``data`` is empty at times without a sampled point);
  * scan.png with all panels.
"""

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c13model import ACETATE_TRACERS, Parameters, activity_scan, add_gaussian_noise, fit_activity, simulate, with_activity

FACTORS = {"pdh": [0.25, 0.5, 0.75, 1.0], "ldh": [0.25, 0.5, 1.0, 2.0]}  # relative to baseline
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark = low -> high activity
# (observable column, panel title, y-axis label) for each (substrate, enzyme)
PANELS = {
    ("glucose", "pdh"): [
        ("Glu_C4_FE", "Glutamate C4", "Glu C4 fractional enrichment"),
        ("Glu_C3_FE", "Glutamate C3", "Glu C3 fractional enrichment"),
        ("Glu_C4_D45", "Glutamate C4 multiplet", "Glu C4 D45 fraction of C4 signal"),
    ],
    ("glucose", "ldh"): [
        ("Lac_C3_FE", "Lactate C3", "Lac C3 fractional enrichment"),
        ("Glu_C4_FE", "Glutamate C4", "Glu C4 fractional enrichment"),
        ("Glu_C3_FE", "Glutamate C3", "Glu C3 fractional enrichment"),
    ],
    ("acetate", "pdh"): [
        ("Glu_C4_FE", "Glutamate C4", "Glu C4 fractional enrichment"),
        ("Glu_C3_FE", "Glutamate C3", "Glu C3 fractional enrichment"),
        ("Gln_C4_FE", "Glutamine C4", "Gln C4 fractional enrichment"),
    ],
}


def filename_from_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="glucose", choices=["glucose", "acetate"])
    ap.add_argument("--enzyme", default="pdh", choices=["pdh", "ldh"])
    ap.add_argument("--mode", default=None,
                    help="pdh: compensated|uncompensated (default compensated for glucose, uncompensated "
                         "for acetate); ldh: both|exchange|net (default both)")
    ap.add_argument("--compensate-with", default="dil", choices=["dil", "acetate"])
    ap.add_argument("--acetate-tracer", default="2-13C", choices=sorted(ACETATE_TRACERS))
    ap.add_argument("--V-ac", type=float, default=0.15, help="acetate oxidation flux for --substrate acetate")
    ap.add_argument("--noise-sd", type=float, default=0.02, help="absolute SD in fraction units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if (args.substrate, args.enzyme) not in PANELS:
        sys.exit("LDH does not affect acetate-derived labeling in this model: there is no route from the "
                 "TCA cycle back to pyruvate/lactate (malic enzyme/PEPCK are not modeled). "
                 "Use --substrate glucose for the LDH scan.")
    enzyme = args.enzyme
    mode = args.mode or {"pdh": "compensated" if args.substrate == "glucose" else "uncompensated",
                         "ldh": "both"}[enzyme]
    base = Parameters()
    if args.substrate == "acetate":
        base = replace(base, glucose_tracer=None, acetate_tracer=args.acetate_tracer, V_ac=args.V_ac)
    tracer = "[U-13C]glucose" if args.substrate == "glucose" else f"[{args.acetate_tracer}]acetate"
    panels = PANELS[(args.substrate, enzyme)]
    factors = FACTORS[enzyme]
    extra = {"compensate_with": args.compensate_with} if enzyme == "pdh" else {}

    out = Path(__file__).parent / "output" / f"{args.substrate}_{enzyme}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c, _, _ in panels]
    fcol = f"{enzyme}_factor"
    t_fine = np.arange(0, 120.5, 0.5)
    sample_t = np.arange(0, 121, 5.0)

    model = activity_scan(factors, enzyme, base, mode, t_eval=t_fine, **extra)
    measured = add_gaussian_noise(model[model.time_min.isin(sample_t)], args.noise_sd, columns=cols, seed=args.seed)

    fits, fitted = {}, []
    for f in factors:
        fit = fit_activity(measured[measured[fcol] == f][["time_min", *cols]], cols, enzyme, base, mode, **extra)
        fits[f] = fit
        curve = simulate(fit.params, t_eval=t_fine).observables()[cols].reset_index()
        curve.insert(0, fcol, f)
        fitted.append(curve)
        print(f"{enzyme.upper()} x {f:g}: fitted {fit.factor:.3f} ± {fit.factor_se:.3f} "
              f"(residual SD {fit.residual_sd:.4f})")
    fitted = pd.concat(fitted, ignore_index=True)

    keys = [fcol, "time_min"]
    for col, _, ylabel in panels:
        table = (
            model[keys + [col]].rename(columns={col: "simulated"})
            .merge(measured[keys + [col]].rename(columns={col: "data"}), on=keys, how="left")
            .merge(fitted[keys + [col]].rename(columns={col: "fit"}), on=keys)
        )
        table[f"fitted_{fcol}"] = table[fcol].map({f: r.factor for f, r in fits.items()})
        table[f"fitted_{fcol}_se"] = table[fcol].map({f: r.factor_se for f, r in fits.items()})
        table.to_csv(out / filename_from_label(ylabel), index=False, float_format="%.6g")
        print(f"Wrote {out / filename_from_label(ylabel)}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.2), constrained_layout=True)
    for ax, (col, title, ylabel) in zip(axes, panels):
        for f, color in zip(factors, COLORS):
            m = model[model[fcol] == f]
            s = measured[measured[fcol] == f]
            ft = fitted[fitted[fcol] == f]
            ax.plot(m.time_min, m[col], color=color, lw=2, label=f"{enzyme.upper()} × {f:g}")
            ax.plot(ft.time_min, ft[col], color=color, lw=1.5, ls="--")
            ax.scatter(s.time_min, s[col], s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.tick_params(colors="#55544f")
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    handles += [plt.Line2D([], [], color="#55544f", lw=2), plt.Line2D([], [], color="#55544f", lw=1.5, ls="--"),
                plt.Line2D([], [], color="#55544f", marker="o", ls="", ms=4)]
    labels += ["simulated", "fit", "data (+ noise)"]
    fig.legend(handles, labels, frameon=False, loc="outside right upper")
    fig.suptitle(
        f"{tracer}, {enzyme.upper()} scan ({mode}; Gaussian noise SD {args.noise_sd:g}; "
        f"{enzyme.upper()} fitted jointly to all panels)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.savefig(out / "scan.png", dpi=150)
    print(f"Wrote {out / 'scan.png'}")


if __name__ == "__main__":
    main()
