"""Willing-and-able model under the COVID shocks: compatibility map and best fit (plan Steps 7 and 9; 30 Oct 2026).

Run (after calibrate_mmp.py; not run yet):
    python 2026-10-30/run_shocks.py                          base: Jan and Mar 2021 on, April 2020 off, pi = 1 (user, 30 Oct)
    python 2026-10-30/run_shocks.py --rounds 2020-04 2021-01 2021-03 --tag all-rounds   all three rounds (robustness)
    python 2026-10-30/run_shocks.py --pi 0                   no stimulus effect
    --kappas / --betas  subsets of the grids;  --quick  kappa {0, 0.5, 0.8, 1}, beta step 0.1;  --tag NAME
Reads outputs/calibration.csv (normal times calibrated at kappa = 0; kappa enters only here, in both runs). For each
kappa and each beta (beta_fit.csv also reports cf_*_vs_k0: the no-COVID level at kappa vs kappa = 0):
    COVID run: CPS employment 2019-01..2022-12, stimulus in the rounds switched on, crisis N*_t (cumulative
               excess deaths, run_household_model.need_stock); no-COVID run: same calendar month of 2019,
               no stimulus, N* = 0.
    index = expected transfers COVID / no-COVID, quarterly, against Banxico counts / 2017-19 trend
            (run_household_model.banxico_target); RMSE 2020Q2..2021Q4, 2022 held out; spikes Jan/Mar 2021.
    beta is also fitted continuously for each kappa (bounded scalar search on the quarterly RMSE).
ENIGH comparisons (Step 9 tests 2-3), per cell:
    receipt index 2022 (mean 2022Q1-Q3 receipt COVID / no-COVID; ENIGH: receiving households -0.02 vs trend),
    transfers per receiving household index 2022 (ENIGH +0.09 vs trend),
    Mar->Apr 2020 log change in receipt, COVID minus no-COVID (ENIGH 2020 minus 2018: +0.016, s.e. 0.048).
Outputs (aggregates) in outputs/shocks[-TAG]/: map.csv, beta_fit.csv, quarterly_paths.csv, monthly_paths.csv,
run_settings.json. beta_fit.csv also reports, at the fitted beta, how often the reserve runs out (share of
households short of reserve migrants, share of drawn replacements left unfilled; 2020Q2 and Apr 2020-Dec 2021).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import willing_able as wa
import run_household_model as hm                 # DATES, FIT_Q, HOLD_Q, quarterly, log_rmse, spikes, targets

KAPPAS = [round(k, 1) for k in np.arange(0, 1.01, 0.1)]
BETAS = np.round(np.arange(0.0, 1.201, 0.05), 2)
BETA_MAX = 3.0
TAU_C = 0.04
ENIGH = dict(receipt_vs_trend_2018_22=-0.02, tpr_vs_trend_2018_22=0.09, mar_apr_2020_minus_2018=0.016)


class Shocks:
    def __init__(self, pop, cal, rounds, pi):
        self.pop, self.cal, self.rounds, self.pi = pop, cal, rounds, pi
        dates = hm.DATES
        self.cf_dates = pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in dates])
        self.covid = wa.abm.make_inputs(dates, pop['rate_columns'])
        self.cf = wa.abm.make_inputs(self.cf_dates, pop['rate_columns'])
        self.a_covid = wa.ability(pop, self.covid, pi, wa.stimulus_indicator(dates, rounds))
        self.a_cf = wa.ability(pop, self.cf)
        self.need = hm.need_stock(dates)
        self.target, self.bx_info = hm.banxico_target()
        bx_m, trend_m, _ = hm.e1.banxico()
        self.spike_bx = hm.spikes((bx_m / trend_m).reindex(dates))
        self._cf = {}

    def counterfactual(self, kappa):
        """No-COVID run (independent of beta): (T, H) receipt and transfers."""
        if kappa not in self._cf:
            p = wa.params_for(self.cal, kappa)
            d = wa.desire(self.pop, len(hm.DATES), p['a_c'], p['a_k'], p['eta'], g=p['g'])
            self._cf[kappa] = wa.receipt_and_transfers(self.pop, wa.StateTables(self.pop, self.a_cf, d).pairs(kappa),
                                                       p['b'])
        return self._cf[kappa]

    def run(self, kappa, beta):
        p = wa.params_for(self.cal, kappa)
        d = wa.desire(self.pop, len(hm.DATES), p['a_c'], p['a_k'], p['eta'], beta, self.need, p['g'])
        Pc, Tc = wa.receipt_and_transfers(self.pop, wa.StateTables(self.pop, self.a_covid, d).pairs(kappa), p['b'])
        Pf, Tf = self.counterfactual(kappa)
        s = lambda z: pd.Series(z.sum(1), index=hm.DATES)
        return s(Pc), s(Tc), s(Pf), s(Tf)

    def summary(self, kappa, beta):
        rc, tc, rf, tf = self.run(kappa, beta)
        q = hm.quarterly(tc) / hm.quarterly(tf)                              # transfer index
        qr = hm.quarterly(rc) / hm.quarterly(rf)                             # receipt index
        sp = hm.spikes(tc / tf)
        q22 = [x for x in hm.HOLD_Q if x <= '2022Q3']
        lr = np.log(rc / rf)
        row = dict(kappa=kappa, beta=beta, rmse_fit=hm.log_rmse(q, self.target, hm.FIT_Q),
                   rmse_holdout_2022=hm.log_rmse(q, self.target, hm.HOLD_Q),
                   index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                   mean_index_2022=float(q[hm.HOLD_Q].mean()),
                   spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                   spike_rmse=float(np.sqrt(np.mean([(sp[m] - self.spike_bx[m]) ** 2 for m in sp]))),
                   receipt_index_2022=float(qr[q22].mean()),
                   tpr_index_2022=float((q[q22] / qr[q22]).mean()),
                   mar_apr_2020_receipt=float(lr['2020-04-01'] - lr['2020-03-01']))
        row['count_compatible'] = row['rmse_fit'] <= TAU_C
        paths = pd.DataFrame({'quarter': q.index, 'kappa': kappa, 'beta': beta, 'index_transfers': q.values,
                              'index_receipt': qr.values, 'index_banxico': self.target[q.index].values})
        monthly = pd.DataFrame({'date': hm.DATES, 'kappa': kappa, 'beta': beta, 'index_transfers': (tc / tf).values,
                                'index_receipt': (rc / rf).values})
        return row, paths, monthly

    def level_gap(self, kappa):
        """No-COVID run at kappa vs kappa = 0: how far stepping-in for normal 2019 job losers moves the level away
        from the kappa = 0 calibration (the index is a ratio, so most of it cancels)."""
        rf, tf = (self.counterfactual(kappa)[i].sum() for i in (0, 1))
        r0, t0 = (self.counterfactual(0.0)[i].sum() for i in (0, 1))
        return dict(cf_receipt_vs_k0=float(rf / r0), cf_transfers_vs_k0=float(tf / t0))

    def shortage(self, kappa, beta):
        """How often stepping-in is capped by the reserve (decision 2: core roles first), COVID run."""
        p = wa.params_for(self.cal, kappa)
        d = wa.desire(self.pop, len(hm.DATES), p['a_c'], p['a_k'], p['eta'], beta, self.need, p['g'])
        drawn, unfilled, short = wa.StateTables(self.pop, self.a_covid, d).shortage(kappa)
        out = {}
        for name, m in (('2020q2', (hm.DATES >= '2020-04-01') & (hm.DATES <= '2020-06-01')),
                        ('2020_21', (hm.DATES >= '2020-04-01') & (hm.DATES <= '2021-12-01'))):
            out[f'share_households_short_{name}'] = float(short[m].mean() / self.pop['H'])
            out[f'share_replacements_unfilled_{name}'] = float(unfilled[m].sum() / max(drawn[m].sum(), 1e-12))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', nargs='*', default=['2021-01', '2021-03'])   # base: April 2020 off (user, 30 Oct)
    ap.add_argument('--pi', type=float, default=1.0)
    ap.add_argument('--kappas', type=float, nargs='*', default=None)
    ap.add_argument('--betas', type=float, nargs='*', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    kappas = args.kappas or ([0.0, 0.5, 0.8, 1.0] if args.quick else KAPPAS)
    betas = np.asarray(args.betas) if args.betas else (np.round(np.arange(0, 1.201, 0.1), 2) if args.quick else BETAS)
    out = wa.OUT / ('shocks' + (f'-{args.tag}' if args.tag else ''))
    out.mkdir(parents=True, exist_ok=True)

    pop = wa.load_population()
    cal = wa.load_calibration()
    sh = Shocks(pop, cal, args.rounds, args.pi)
    rows, paths, monthly, fits = [], [], [], []
    for kappa in kappas:
        for beta in betas:
            r, p, m = sh.summary(kappa, float(beta))
            rows.append(r), paths.append(p)
            if beta == 0:
                monthly.append(m)
        res = minimize_scalar(lambda b: sh.summary(kappa, b)[0]['rmse_fit'], bounds=(0.0, BETA_MAX),
                              method='bounded', options=dict(xatol=1e-3))
        r, p, m = sh.summary(kappa, float(res.x))
        # theta_core_*: desire of a core migrant who is the household's only core migrant (g log 1 = 0)
        fits.append(dict(r, **sh.level_gap(kappa), **sh.shortage(kappa, float(res.x)),
                         theta_core_2019=float(wa.expit(wa.params_for(cal, kappa)['a_c'])),
                         theta_core_2021_12=float(wa.expit(wa.params_for(cal, kappa)['a_c'] + res.x))))
        paths.append(p.assign(fitted=True)), monthly.append(m.assign(fitted=True))
        print(f'kappa {kappa}: beta {res.x:.3f}, rmse {res.fun:.4f}', flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(out / 'map.csv', index=False)
    pd.DataFrame(fits).to_csv(out / 'beta_fit.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / 'quarterly_paths.csv', index=False)
    pd.concat(monthly, ignore_index=True).to_csv(out / 'monthly_paths.csv', index=False)
    best = grid.loc[grid.rmse_fit.idxmin()]
    settings = dict(
        plan='2026-10-30/model_plan.tex, Steps 1-7 and 9', stimulus_rounds=args.rounds, pi=args.pi,
        kappas=list(map(float, kappas)), betas=list(map(float, betas)), tau_C=TAU_C,
        need='cumulative excess deaths since Mar 2020 (run_household_model.need_stock), 1 in Dec 2021',
        target='Banxico transactions / 2017-19 trend (log-linear, month effects), quarterly',
        transfers='T = E[s(E) max(E^c + E^k, 1)], one transfer per effective sender',
        enigh_reference=ENIGH, banxico_trend=sh.bx_info,
        spike_banxico={k: round(v, 4) for k, v in sh.spike_bx.items()},
        best_cell=dict(kappa=float(best.kappa), beta=float(best.beta), rmse=float(best.rmse_fit)),
        n_compatible=int(grid.count_compatible.sum()))
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    print(grid.pivot(index='beta', columns='kappa', values='rmse_fit').round(3).to_string())


if __name__ == '__main__':
    main()
