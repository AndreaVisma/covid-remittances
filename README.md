# COVID-19 and remittances to Mexico: what kept the money flowing?

In April 2020 the World Bank expected remittances to fall by about a fifth. Remittances to Mexico rose instead. Transfer counts stayed on their pre-pandemic trend in 2020, then ended 2021 about 11% above it, and stayed high afterwards. This repository holds one chapter of a PhD on remittance networks in crises. It asks which mechanisms, acting through the real networks between Mexican households and their migrants in the US, can explain this path and which can't.

**Status (30 Oct 2026):** work in progress. A paper draft exists (`paper/`), built on the current base model. A revised model, the "willing-and-able" model, is calibrated, but its shock runs have not been run yet.

## The idea in brief

- **Networks:** we take the real household–migrant networks of the Mexican Migration Project (MMP): 3,405 households interviewed 2010–19 and their 7,999 links to migrants in the US. Links are core (head, spouse or child), close kin or distant kin.
- **Shock:** each month every migrant is employed or not, following US employment rates by occupation (CPS). US stimulus checks let out-of-work migrants send in the months they arrive.
- **Mechanisms tested:**
  - **Stepping-in (κ):** when a sender loses their job, another migrant linked to the household sends instead.
  - **Need:** more of each household's migrants send as Mexico's crisis builds up, measured by cumulative excess deaths.
  - **Benefits:** US stimulus checks (π).
- **Target:** the model gives monthly household receipt and transfers. These are compared with Banco de México transfer counts relative to their 2017–19 trend. 2020Q2–2021Q4 is the fit window, and 2022 is held out.
- **Outside checks:** the ENIGH household survey (2016, 2018, 2020, 2022), plus state and municipal panels of remittances against excess mortality.

## Main findings so far (current base model)

1. **The puzzle is a 2021 level shift, not growth in 2020.** Counts are flat against the trend in 2020. The March 2020 spike and April dip are a timing shift: migrants sent early at the weak peso, and March plus April together are exactly on trend.
2. **Job loss alone implies a clear dip in 2020Q2; the data show almost none.** Matching the data needs near-complete replacement of lost senders by other migrants linked to the same household. Out-of-work migrants who keep sending (from savings, say) fit the national series equally well. The two can't be separated with aggregate data.
3. **The 2021 rise came from more transfers per receiving household, not from more receiving households (ENIGH).** In the model, more migrants per household send as the national crisis accumulates, and this closes about 80% of the 2021Q4 gap. The rise was common to all regions: state and municipal panels show no gradient with local mortality.
4. **Stimulus checks moved transfers between months** (Jan and Mar 2021) without raising quarterly totals.

## Where to look

| Folder | What it is |
|---|---|
| **`2026-10-28/handoff.md`** | **Start here.** Current state, all decisions and numbers, how to resume, paper outline. |
| **`2026-10-30/`** | The newest model ("willing-and-able"): each migrant is *able* to send (employed, or got a stimulus check) and may *want* to send. Able migrants who want to send are called **able senders**. Contains the plan (`model_plan.tex`), the 2019 calibration and its write-up (`calibration_2019.tex`), the figures, and the scripts for the shock runs, not yet run. See its `README.md`. |
| **`network-compatibility-model/`** | The current base model and everything around it. `run_all.py` rebuilds every output (~75 min). `model-specification.md` is the formal model. `data-prep/` builds the inputs (MMP population, CPS employment, benefits, CPI). `recipient-level/` holds the household model, robustness checks and figures. `identification/` holds the ENIGH, state and municipal evidence. `handoff.md` is the archive of earlier decisions. |
| **`paper/`** | Paper draft (`main.tex` → `main.pdf`, 29 Oct), with `refs.bib` and the figure script. |
| `oct-26/` | Research plans from early October (the three-month plan governs scope) and earlier analyses. |
| `analysis/`, `scripts/` | First descriptive look at remittances to Mexico, Pakistan and the Philippines (Sept 2026). |
| `mexico/`, `simulation/`, `microsimulation/` | Early prototypes, superseded by the network model. |
| `literature/` | Bibliography. |

## Running the code

Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib` and `statsmodels`. Run scripts from the repository root, for example:

```
python network-compatibility-model/run_all.py --skip-robustness
python 2026-10-30/check_special_cases.py
```

## Data

The raw data are **not in this repository** and must not be committed: MMP and IPUMS-CPS (licensed), ENIGH, Banxico and FRED downloads. They are expected under `data/`, which is git-ignored. Only aggregate outputs are kept here. To rebuild the inputs you need your own access to MMP and IPUMS; then run `run_all.py --data-prep`.

## Contact

Andrea Vismara, Complexity Science Hub Vienna (vismara@csh.ac.at).
