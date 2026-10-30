# Comparison model: the colleague's specification (no stepping-in)

Script: `recipient-level/robustness/colleague_spec.py` (~10 min). Outputs: `map.csv` (every variant x Covid
version x B x A mode, including the 2019 profile columns), `summary.csv` (best B by quarterly RMSE per variant x
Covid version x A mode), `quarterly_paths.csv`, `run_settings.json`, `colleague_spec.png`.

## Specification

- Keen core migrant: `p_j,t = e expit(A + B + C Covid_t) + (1 - e) expit(A + C Covid_t)`, e = CPS rate of the
  migrant's occupation (`abm.able_probability`, Params(0, 0)). Out-of-work / employed keen ratio = expit(A) / expit(A + B).
- Receipt: `E[expit(alpha + beta_k Keen + delta FarFam (+ eta' need))]`, Keen Poisson-binomial over core migrants
  (exact), FarFam = number of kin links (fixed, not gated). No stepping-in, **no count mapping**: the index is on
  expected receiving households and is compared with Banxico counts directly.
- Covid_t: **E** = `hm.need_stock` (cumulative excess deaths, 1 in Dec 2021); **S** = `abm.salience_series`
  (monthly reported deaths, flow). 0 in 2019 and in the no-COVID run.
- Variants (cumulative): (i) literal; (ii) + stimulus (e -> e + (1 - e) pi_t in Jan and Mar 2021, COVID run);
  (iii) = (ii) + household need features (no_worker, any_older) in the receipt logit.
- 2019: ML on MMP receipt (3,252 households, `e0.target_mask`), 2019 CPS, Covid = 0, per B in
  {0, 0.5, 1, 2, 3, 4, 6, 10}. C >= 0 fitted per row (bounded scalar, quarterly log RMSE 2020Q2-2021Q4); 2022 held out.

## 2019 calibration

- **A is identified** (SE 0.06 at every B), but only because **beta_k runs to its bound (18-20)**: one keen core
  migrant makes receipt certain, so the receipt equation collapses to the migrant-level noisy-OR
  `1 - (1 - expit(alpha + delta FarFam)) prod_j (1 - p_j)` (M1 with a kin floor). The mean core keen probability
  is 0.41 (M1: 0.44). Pinning A to the M1 value costs 2 log-lik units and changes nothing below (`A_mode = pinned`).
- **Profile over B**: monotone, maximum at B = 0; Delta loglik -0.46 (B = 0.5), -0.88 (1), -1.48 (2), -1.99 (10).
  The pure employment gate is weakly disfavoured (LR 4.0), B <= 2 is not. The cross-section does not pin B.
- **Against the base receipt logit** (kappa = 0, 7 parameters, loglik -1128.27, AIC 2270.5): colleague (i)/(ii)
  -1185.1 (5 incl. B, AIC 2380.2, +110); (iii) -1169.6 (7, AIC 2353.2, +83). The gap is mostly the kin effect:
  the base lets kin matter only without a core migrant; FarFam is linear for every household (delta 0.21).

## Results (A free; Banxico 2020Q2 0.974, 2021Q4 1.108, 2022 mean 1.099, spikes +4.1 / +3.6)

| variant | Covid | best B | out-of-work / employed keen | C | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | 2022 mean | spikes Jan / Mar | 2019 loglik / AIC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (i) literal | E | 1 | 0.50 | 0.33 | 0.0098 | 0.024 | 0.976 | 1.093 | 1.104 | +1.2 / +0.5 | -1186.0 / 2382.0 |
| (i) literal | S | 10 | 0.00 | 0.13 | 0.0426 | 0.091 | 0.981 | 1.018 | 1.010 | +6.2 / +0.2 | -1187.1 / 2384.2 |
| (ii) + stimulus | E | 1 | 0.50 | 0.31 | 0.0130 | 0.024 | 0.976 | 1.090 | 1.100 | +3.7 / +2.5 | -1186.0 / 2382.0 |
| (ii) + stimulus | S | 1 | 0.50 | 0.09 | 0.0455 | 0.093 | 1.003 | 1.013 | 1.007 | +6.6 / +1.9 | -1186.0 / 2382.0 |
| (iii) + need | E | 1 | 0.50 | 0.32 | 0.0129 | 0.024 | 0.976 | 1.090 | 1.100 | +3.6 / +2.4 | -1170.5 / 2354.9 |
| (iii) + need | S | 2 | 0.21 | 0.10 | 0.0454 | 0.092 | 0.989 | 1.014 | 1.008 | +8.9 / +3.3 | -1171.1 / 2356.1 |
| *base, kappa 1, beta 0.65* | E (on senders) | | | | 0.0123 | 0.024 | 0.975 | 1.093 | 1.106 | +4.1 / +2.9 | -1128.3 (kappa 0) / 2270.5 |

Variant (ii), Covid E, across B (job loss only, C = 0 -> C fitted):

| B | out-of-work / employed | 2020Q2 at C = 0 | 2020Q2 | RMSE fit | spikes Jan / Mar | spike RMSE |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.000 | 1.007 | 0.0174 | +0.4 / +0.0 | 3.7 |
| 0.5 | 0.73 | 0.983 | 0.990 | 0.0135 | +2.1 / +1.4 | 2.1 |
| 1 | 0.50 | 0.969 | 0.976 | 0.0130 | +3.7 / +2.5 | 0.9 |
| 2 | 0.22 | 0.949 | 0.956 | 0.0168 | +5.7 / +4.1 | 1.2 |
| 3 | 0.08 | 0.940 | 0.947 | 0.0198 | +6.8 / +4.8 | 2.1 |
| 10 (gate) | 0.00 | 0.934 | 0.941 | 0.0220 | +7.4 / +5.3 | 2.6 |

(B = 10 at C = 0 gives 0.934, close to the base without stepping-in, 0.930.)

## Reading

1. **2020Q2 without stepping-in:** yes, with B ~ 1: out-of-work migrants keep sending at half the employed rate
   (expit(A) / expit(A + B) = 0.50). Finite B does the job of kappa: both say most lost sending is not lost.
   The pure gate (B large) gives 0.94, as the base at kappa = 0.
2. **2021 rise and 2022:** with the cumulative stock (E), C ~ 0.31-0.33 on the keen logit fits 2020Q2-2021Q4 as well
   as the base (RMSE 0.010-0.013) and the 2022 holdout (0.024). With the flow (S) it fails (RMSE 0.043-0.046,
   2022 holdout 0.09, 2021Q4 1.01-1.02): as in the base (need_series_decay), the 2021-22 level needs a non-decaying series.
3. **Spikes:** B behaves like kappa. More cushioning (smaller B) leaves less lost sending for the checks to restore,
   so smaller spikes: B = 1 gives +3.7 / +2.5 (base kappa 1: +4.1 / +2.9), the gate +7.4 / +5.3 (base kappa 0: +7.8 / +5.8).
   Spikes and the trough both point to B ~ 1; they do not discriminate B from kappa. Without the stimulus (i) the
   quarterly fit is best (0.0098) but the spikes are missing, as in stimulus_gamma.
4. **Where need lands:** the colleague's model has no count mapping, so the whole 2021-22 rise is in the number of
   receiving households: receipt 2022 +10% (2022Q3 index 1.099, +9.4 log points) against ENIGH -0.02 (s.e. 0.10)
   vs trend, with transfers per receiving household +0.09. About one s.e. off and the wrong margin: ENIGH puts the
   rise on the intensive margin, where the base (receipt 2022 index 1.001; need through senders per receiving
   household) puts it.
5. **2019:** the colleague's receipt equation fits 2019 MMP receipt clearly worse than the base logit
   (Delta AIC +83 to +110), mostly because of the linear kin term; its 2019 optimum is the migrant-level noisy-OR
   (beta_k -> infinity), not a smooth receipt logit.
