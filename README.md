# c13model: two-compartment ¹³C labeling of glutamate and glutamine vs. PDH and LDH activity

A Python simulation of ¹³C labeling in a **neuron + astrocyte** brain model during an infusion
of **[U-¹³C]glucose** or **¹³C-acetate**. You can vary two enzyme activities, separately:
* **pyruvate dehydrogenase (PDH)**;
* **lactate dehydrogenase (LDH)**.

Each can be changed in neurons, astrocytes or both. Gaussian noise can be added to the output
to make synthetic NMR data, and each activity can be fitted back from noisy data.

## References

* **Model structure (neuron/astrocyte, glutamate-glutamine cycle):** Shen J. *Modeling the
  glutamate–glutamine neurotransmitter cycle.* Front Neuroenergetics. 2013;5:1.
  doi:[10.3389/fnene.2013.00001](https://doi.org/10.3389/fnene.2013.00001)
* **Original single-compartment model:** Mason GF, Rothman DL, Behar KL, Shulman RG. *NMR
  determination of the TCA cycle rate and α-ketoglutarate/glutamate exchange rate in rat
  brain.* J Cereb Blood Flow Metab. 1992;12(3):434-447.
  doi:[10.1038/jcbfm.1992.61](https://doi.org/10.1038/jcbfm.1992.61)

> **Note:** the full text of Shen (2013) couldn't be retrieved from the build environment
> (the network policy blocks the publisher and PubMed Central). The model below follows the
> standard two-compartment formulation that paper reviews. **Check the flux definitions and
> the parameter values against the paper before quantitative use.** Every default parameter is
> an illustrative value and is not taken from the paper.

Every pool carries its **full isotopomer distribution** (all 2ⁿ labeling patterns). Multiply
labeled tracers and NMR multiplets (e.g. the C4–C5 doublet) are therefore simulated exactly.

## Model

```
                     glucose (FE_glc(t))                          blood lactate (FE_lac(t))
                   V_gly_n │        │ V_gly_g                           │ V_mct_in
                           ▼        ▼                                   ▼
   NEURON           Pyr_n ◄──LDH_n──► Lac (tissue) ◄──LDH_g──► Pyr_g         ASTROCYTE
                      │ V_pdh_n                                 │ V_pdh_g   │ V_pc
  unlabeled ─V_dil_n─►AcCoA_n                  acetate ─V_ac─► AcCoA_g ◄─V_dil_g─ unlabeled
                      │ V_TCA_n                                 │ V_TCA_g   │
          Asp_n ◄Vx►OAA_n ─► aKG_n ◄─Vx_n─► Glu_n    Glu_g ◄─Vx_g─► aKG_g ◄─ OAA_g ◄┘
                                               │  ▲ V_cyc    │ │ GS: V_cyc + V_pc
                                               │  └──────────┼─┼──── Gln ──V_pc──► efflux
                                               └──V_cyc──────┘ └────► Gln ◄─V_gln_dil─► unlabeled
```

| Neuron | Astrocyte | Intercellular |
|---|---|---|
| V_TCA_n = V_pdh_n + V_dil_n | V_TCA_g = V_pdh_g + V_ac + V_dil_g | V_cyc: Glu_n → Glu_g and Gln → Glu_n |
| V_x_n: aKG_n ⇄ Glu_n, OAA_n ⇄ Asp_n | V_x_g: aKG_g ⇄ Glu_g | glutamine synthesis V_GS = V_cyc + V_pc |
| LDH_n: Pyr_n ⇄ Lac | V_pc: Pyr_g → OAA_g (balanced by glutamine efflux) | V_gln_dil: glutamine isotopic dilution |
| | LDH_g: Pyr_g ⇄ Lac | V_mct_in: blood lactate exchange |

Main modeling choices:
* Acetate is taken up and oxidised only by astrocytes.
* Pyruvate carboxylase is astrocytic.
* Glutamine is an astrocytic pool.
* NMR sees **total glutamate**. It is reported as `Glu`, the pool-weighted mix of `Glu_n` and
  `Glu_g`. The compartments are also reported separately.

Carbon transitions:
* PDH: acetyl C1 ← Pyr C2, acetyl C2 ← Pyr C3.
* Acetate: acetyl C1 ← acetate C1, acetyl C2 ← acetate C2.
* Citrate synthase → IDH: α-KG C1..C5 ← OAA C4, C3, C2, acetyl C2, acetyl C1.
* α-KG dehydrogenase loses C1. Succinate is symmetric, so the label is scrambled 50/50 on the way back to OAA.
* PC: OAA C1..C3 ← Pyr C1..C3, and OAA C4 ← CO₂.

For each pool of size *P*, `P·dx/dt = Σ V_in·x_in − (Σ V_out)·x`, integrated with
`scipy.solve_ivp` (BDF).

| Pool | Default (µmol/g) | Pool | Default (µmol/g) |
|---|---|---|---|
| Pyr_n / Pyr_g | 0.08 / 0.02 | Glu_n / Glu_g | 9.0 / 1.0 |
| Lac | 1.0 | Gln | 4.0 |
| AcCoA_n / AcCoA_g | 0.02 / 0.005 | Asp_n | 3.0 |
| OAA_n / OAA_g | 0.02 / 0.005 | aKG_n / aKG_g | 0.2 / 0.05 |

Default fluxes (µmol/g/min, all illustrative):
* **Neuron:** V_pdh_n 0.9; V_x_n 57 (unverified placeholder); V_ldh_n 1.0; V_lac_net_n 0.
* **Astrocyte:** V_pdh_g 0.1; V_dil_g 0.05; V_pc 0.05; V_x_g 57; V_ldh_g 0.5; V_lac_net_g 0.05.
* **Intercellular:** V_cyc 0.25; V_gln_dil 0; V_mct_in 0.1.

### Tracers

* `glucose_tracer` accepts `"U-13C"` (default), `"1-13C"`, `"1,6-13C"`, `"2-13C"`, `"6-13C"` or `None`.
* `acetate_tracer` accepts `"1-13C"`, `"2-13C"`, `"1,2-13C"` or `None`. It needs `V_ac > 0`.
* Enrichment inputs `glucose_fe`, `acetate_fe` and `lactate_fe` are passed to `simulate`:
  * The glucose and acetate defaults are 0.7·(1−e^(−t/1 min)).
  * Blood lactate defaults to 0.
  * Measured curves can be passed with `tabulated_enrichment`.

### PDH activity: `with_pdh_activity(base, factor, mode, compartment, compensate_with)`

* `compartment`: `"both"` (default), `"neuron"` or `"astrocyte"`.
* `mode="compensated"`: each compartment's V_TCA stays fixed. The lost PDH flux is replaced by
  unlabeled substrate (V_dil), or in astrocytes by acetate (`compensate_with="acetate"`).
* `mode="uncompensated"`: V_TCA falls along with PDH.
* V_cyc and V_pc are unchanged in both modes.

### LDH activity: `with_ldh_activity(base, factor, mode, compartment)`

* `mode="both"`: forward and reverse LDH fluxes both scale, as when the amount of enzyme
  changes. Net lactate production and glycolysis scale too.
* `mode="exchange"`: only the Pyr ⇄ Lac exchange scales.
* `mode="net"`: only net lactate production scales.

## Usage

```bash
pip install -e .[test]
pytest
python examples/scan.py                                  # [U-13C]glucose, PDH in both compartments
python examples/scan.py --compartment astrocyte          # PDH changed in astrocytes only
python examples/scan.py --enzyme ldh                     # [U-13C]glucose, LDH
python examples/scan.py --enzyme ldh --fit-lactate       # ... also fit lactate C3
python examples/scan.py --substrate acetate              # [2-13C]acetate, PDH
python examples/scan.py --timepoint 30 --noise-sd 0.03
```

```python
from c13model import Parameters, simulate, add_gaussian_noise, with_pdh_activity, with_ldh_activity, fit_pdh

p = with_pdh_activity(Parameters(), 0.5, "compensated", compartment="astrocyte")
obs = simulate(p, t_end=120).observables()
obs[["Glu_C4_FE", "Gln_C4_FE", "Glu_n_C4_FE", "Glu_g_C4_FE", "Lac_C3_FE"]]
noisy = add_gaussian_noise(obs, sd=0.02, seed=1)
fit = fit_pdh(noisy.reset_index(), ["Glu_C4_FE", "Gln_C4_FE"], compartment="astrocyte")
```

`observables()` gives the following for `Glu` (total), `Gln`, `Glu_n`, `Glu_g`, `Asp_n` and
`Lac` at each carbon *k*:
* `*_Ck_FE`: fractional enrichment.
* `*_Ck_conc`: ¹³C concentration in µmol/g.
* Multiplet fractions: `*_Ck_S`, `*_Ck_D..` and `*_Ck_Q`.

### Example script outputs

`examples/scan.py` plots **glutamate and glutamine separately**, as `glutamate.png` and
`glutamine.png`. Each figure has three panels:
1. C4 fractional enrichment over time at 4 activity levels.
2. C3 fractional enrichment over time at 4 activity levels.
3. C4 fractional enrichment at `--timepoint` (default 60 min) **plotted against the activity**.

Solid lines are the simulation, dots are simulation + Gaussian noise, and dashed lines / open
diamonds are the fit. The activity is fitted once per level, jointly to all four time-course
panels.

**One CSV is written per plot panel, named from its y-axis label:**

| y-axis label | CSV | Columns |
|---|---|---|
| Glu C4 fractional enrichment | `Glu_C4_fractional_enrichment.csv` | `{enzyme}_factor, time_min, simulated, data, fit, fitted_{enzyme}_factor, fitted_{enzyme}_factor_se` |
| Glu C3 fractional enrichment | `Glu_C3_fractional_enrichment.csv` | same |
| Glu C4 fractional enrichment at 60 min | `Glu_C4_fractional_enrichment_at_60_min.csv` | `{enzyme}_factor, simulated, data, fit, fitted_{enzyme}_factor, fitted_{enzyme}_factor_se` |
| Gln C4 fractional enrichment | `Gln_C4_fractional_enrichment.csv` | as Glu |
| Gln C3 fractional enrichment | `Gln_C3_fractional_enrichment.csv` | as Glu |
| Gln C4 fractional enrichment at 60 min | `Gln_C4_fractional_enrichment_at_60_min.csv` | as Glu |

How the CSV cells are filled:
* `data` is filled only at the sampled points (every 5 min, or the 4 levels in the activity panel).
* `fit` is filled only where a fitted value exists.

Outputs from the default runs (seed 0, noise SD 0.02) are in `docs/<substrate>_<enzyme>_<mode>_<compartment>/`.

To plot the CSV files themselves, run `python examples/plot_csv.py <folder or csv> ...` (for example, `python examples/plot_csv.py docs/*/`). It writes one PNG next to each CSV, with the same name and the y-axis label taken from the filename. It also writes an `overview.png` per folder, with glutamate on the top row and glutamine on the bottom.

## What the example runs show

* **PDH, glucose:** glutamate and glutamine C4 fall almost in proportion to PDH activity. Figures below are the change in C4 FE at 60 min as PDH drops from 1× to 0.25×:
  * **Astrocytes only:** mostly moves glutamine (Gln −0.13, Glu −0.04).
  * **Neurons only:** moves glutamate most (Glu −0.43). Glutamine still falls substantially
    (−0.27), because neuronal glutamate feeds glutamine through the cycle. The fits recover the activity to about ±1–2 %.
* **PDH, acetate:** acetate enters astrocytes, so glutamine labels faster and higher than
  glutamate. Lowering PDH raises acetate labeling, because acetate then supplies a larger share
  of acetyl-CoA.
* **LDH, glucose:** LDH mostly sets lactate labeling. Glutamate and glutamine change by less than
  0.02 FE over 0.1–4× LDH, because only the small blood-lactate dilution feeds through. With noise
  SD 0.02, **LDH is not identifiable from glutamate/glutamine alone**; for example, the 1× level
  fits as 1.6 ± 1.6. Adding lactate C3 (`--fit-lactate`) recovers it, for example 0.93 ± 0.09.
  The activity panel's y-axis spans at least 0.2 FE, so the small effect isn't magnified.

## Limitations

* The equations and parameters have not been checked against the full text of Shen (2013).
* The model has no GABAergic compartment, no neuronal glutamine pool and no route from the TCA
  cycle back to pyruvate (malic enzyme/PEPCK). As a result, acetate label never reaches lactate,
  and LDH has no effect on acetate-derived labeling.
* Each fit estimates only one activity factor, with everything else fixed.
