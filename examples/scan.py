"""Two-compartment (neuron/astrocyte) simulation of glutamate and glutamine
13C labeling as a function of PDH or LDH activity, with Gaussian noise and a
fit of the activity back from the noisy data.

Run, for example:
  python examples/scan.py                                    # [U-13C]glucose, PDH (both compartments)
  python examples/scan.py --enzyme ldh                       # [U-13C]glucose, LDH
  python examples/scan.py --compartment astrocyte            # PDH changed in astrocytes only
  python examples/scan.py --substrate acetate                # [2-13C]acetate, PDH

Writes to examples/output/{substrate}_{enzyme}_{mode}_{compartment}[_fit-lactate]/:
  * glutamate.png and glutamine.png (plotted independently), each with
      - C4 and C3 fractional enrichment time courses at 4 activity levels;
      - C4 fractional enrichment at --timepoint vs. activity;
  * one CSV per plot panel, named from the panel's y-axis label:
      time-course panels: {enzyme}_factor, time_min, simulated, data, fit,
                          fitted_{enzyme}_factor, fitted_{enzyme}_factor_se
      activity panel:     {enzyme}_factor, simulated, data, fit,
                          fitted_{enzyme}_factor, fitted_{enzyme}_factor_se
    (``data``/``fit`` are empty where there is no sampled point/fitted level).
The activity is fitted once per level, jointly to all time-course panels of
both figures (optionally adding lactate C3 with --fit-lactate).
"""

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c13model import (
    ACETATE_TRACERS,
    COMPARTMENTS,
    Parameters,
    activity_scan,
    add_gaussian_noise,
    fit_activity,
    simulate,
)
from c13model.fit import _upper_bound

LEVELS = {"pdh": [0.25, 0.5, 0.75, 1.0], "ldh": [0.25, 0.5, 1.0, 2.0]}  # relative to baseline
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark = low -> high activity
INK, MUTED, GRID = "#2f2e2b", "#55544f", "#e5e4e0"
METABOLITES = {"Glu": ("glutamate", "Glutamate (total)"), "Gln": ("glutamine", "Glutamine")}


def filename_from_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="glucose", choices=["glucose", "acetate"])
    ap.add_argument("--enzyme", default="pdh", choices=["pdh", "ldh"])
    ap.add_argument("--mode", default=None,
                    help="pdh: compensated|uncompensated (default compensated for glucose, uncompensated "
                         "for acetate); ldh: both|exchange|net (default both)")
    ap.add_argument("--compartment", default="both", choices=COMPARTMENTS)
    ap.add_argument("--compensate-with", default="dil", choices=["dil", "acetate"])
    ap.add_argument("--acetate-tracer", default="2-13C", choices=sorted(ACETATE_TRACERS))
    ap.add_argument("--V-ac", type=float, default=0.1, help="astrocytic acetate oxidation for --substrate acetate")
    ap.add_argument("--timepoint", type=float, default=60.0, help="time (min) for the activity panel")
    ap.add_argument("--fit-lactate", action="store_true", help="also fit to lactate C3 enrichment")
    ap.add_argument("--noise-sd", type=float, default=0.02, help="absolute SD in fraction units")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.substrate == "acetate" and args.enzyme == "ldh":
        sys.exit("LDH does not affect acetate-derived labeling in this model: there is no route from the "
                 "TCA cycle back to pyruvate/lactate (malic enzyme/PEPCK are not modeled). "
                 "Use --substrate glucose for the LDH scan.")
    enzyme, compartment = args.enzyme, args.compartment
    mode = args.mode or {"pdh": "compensated" if args.substrate == "glucose" else "uncompensated",
                         "ldh": "both"}[enzyme]
    base = Parameters()
    if args.substrate == "acetate":
        base = replace(base, glucose_tracer=None, acetate_tracer=args.acetate_tracer, V_ac=args.V_ac)
    tracer = "[U-13C]glucose" if args.substrate == "glucose" else f"[{args.acetate_tracer}]acetate"
    extra = {"compensate_with": args.compensate_with} if enzyme == "pdh" else {}
    E = enzyme.upper()
    fcol = f"{enzyme}_factor"
    levels = LEVELS[enzyme]
    tp = args.timepoint

    suffix = "_fit-lactate" if args.fit_lactate else ""
    out = Path(__file__).parent / "output" / f"{args.substrate}_{enzyme}_{mode}_{compartment}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    t_fine = np.arange(0, 120.5, 0.5)
    sample_t = np.arange(0, 121, 5.0)
    if tp not in sample_t or tp not in t_fine:
        sys.exit("--timepoint must be a multiple of 5 min between 0 and 120")

    tc_cols = [f"{m}_C{k}_FE" for m in METABOLITES for k in (4, 3)]
    fit_cols = tc_cols + (["Lac_C3_FE"] if args.fit_lactate else [])

    # time courses at the 4 levels, noisy samples, and fits
    model = activity_scan(levels, enzyme, base, mode, compartment, t_eval=t_fine, **extra)
    measured = add_gaussian_noise(model[model.time_min.isin(sample_t)], args.noise_sd, columns=fit_cols,
                                  seed=args.seed)
    fits, fitted = {}, []
    for f in levels:
        fit = fit_activity(measured[measured[fcol] == f][["time_min", *fit_cols]], fit_cols, enzyme, base, mode,
                           compartment, **extra)
        fits[f] = fit
        curve = simulate(fit.params, t_eval=t_fine).observables()[tc_cols].reset_index()
        curve.insert(0, fcol, f)
        fitted.append(curve)
        print(f"{E} x {f:g} ({compartment}): fitted {fit.factor:.3f} ± {fit.factor_se:.3f} "
              f"(residual SD {fit.residual_sd:.4f})")
    fitted = pd.concat(fitted, ignore_index=True)

    # labeling at the chosen time point vs. activity (fine grid)
    upper = _upper_bound(base, enzyme, mode, compartment, args.compensate_with)
    top = min(upper, 1.0) if enzyme == "pdh" and mode == "compensated" else (1.5 if enzyme == "pdh" else 4.0)
    grid = np.unique(np.r_[np.linspace(0.1, top, 19), levels])
    dose = activity_scan(grid, enzyme, base, mode, compartment, t_eval=[0, tp], **extra)
    dose = dose[dose.time_min == tp].reset_index(drop=True)

    keys = [fcol, "time_min"]
    fitted_factor = {f: r.factor for f, r in fits.items()}
    fitted_se = {f: r.factor_se for f, r in fits.items()}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    for met, (fname, name) in METABOLITES.items():
        panels = [
            (f"{met}_C4_FE", f"{name} C4", f"{met} C4 fractional enrichment"),
            (f"{met}_C3_FE", f"{name} C3", f"{met} C3 fractional enrichment"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

        for ax, (col, title, ylabel) in zip(axes, panels):
            table = (
                model[keys + [col]].rename(columns={col: "simulated"})
                .merge(measured[keys + [col]].rename(columns={col: "data"}), on=keys, how="left")
                .merge(fitted[keys + [col]].rename(columns={col: "fit"}), on=keys)
            )
            table[f"fitted_{fcol}"] = table[fcol].map(fitted_factor)
            table[f"fitted_{fcol}_se"] = table[fcol].map(fitted_se)
            table.to_csv(out / filename_from_label(ylabel), index=False, float_format="%.6g")
            print(f"Wrote {out / filename_from_label(ylabel)}")

            for f, color in zip(levels, COLORS):
                t = table[table[fcol] == f]
                ax.plot(t.time_min, t.simulated, color=color, lw=2, label=f"{E} × {f:g}")
                ax.plot(t.time_min, t.fit, color=color, lw=1.5, ls="--")
                ax.scatter(t.time_min, t.data, s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(ylabel)

        # activity panel
        col = f"{met}_C4_FE"
        ylabel = f"{met} C4 fractional enrichment at {tp:g} min"
        ax = axes[2]
        at_tp = measured[measured.time_min == tp].set_index(fcol)[col]
        fit_tp = fitted[fitted.time_min == tp].set_index(fcol)[col]
        table = dose[[fcol, col]].rename(columns={col: "simulated"})
        table["data"] = table[fcol].map(at_tp)
        table["fit"] = table[fcol].map(fit_tp)
        table[f"fitted_{fcol}"] = table[fcol].map(fitted_factor)
        table[f"fitted_{fcol}_se"] = table[fcol].map(fitted_se)
        table.to_csv(out / filename_from_label(ylabel), index=False, float_format="%.6g")
        print(f"Wrote {out / filename_from_label(ylabel)}")

        ax.plot(table[fcol], table.simulated, color=MUTED, lw=2)
        for f, color in zip(levels, COLORS):
            ax.scatter(f, at_tp[f], s=30, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(fitted_factor[f], fit_tp[f], s=40, marker="D", facecolor="none", edgecolor=color,
                       linewidth=1.5, zorder=4)
        ax.set_title(f"C4 at {tp:g} min vs. {E} activity", loc="left", fontsize=11)
        ax.set_xlabel(f"{E} activity (× baseline)")
        ax.set_ylabel(ylabel)
        # keep at least a 0.2 FE span so negligible effects are not magnified
        lo, hi = ax.get_ylim()
        if hi - lo < 0.2:
            mid = (hi + lo) / 2
            ax.set_ylim(mid - 0.1, mid + 0.1)

        for ax in axes:
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED)
        handles, labels = axes[0].get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        handles += [plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=MUTED, lw=1.5, ls="--"),
                    plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=5),
                    plt.Line2D([], [], color=MUTED, marker="D", ls="", ms=5, mfc="none")]
        labels += ["simulated", "fit", "data (+ noise)", f"fit (at fitted {E})"]
        fig.legend(handles, labels, frameon=False, loc="outside right upper")
        fig.suptitle(
            f"{name} labeling from {tracer}: {E} scan ({mode}, {compartment}; noise SD {args.noise_sd:g})",
            x=0.01, ha="left", fontsize=12, color=INK,
        )
        fig.savefig(out / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out / f'{fname}.png'}")


if __name__ == "__main__":
    main()
