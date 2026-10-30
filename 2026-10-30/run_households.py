"""Household paths of the willing-and-able model (plan Step 9 test 4 and Step 10; 30 Oct 2026).

Run (after calibrate_mmp.py and run_shocks.py; not run yet):
    python 2026-10-30/run_households.py                 kappa and beta from outputs/shocks/beta_fit.csv (best fit)
    --kappas 0 0.5 1   kappas to report (beta: the fitted value for each);  --seeds 200;  --shocks-tag NAME
Simulates paired paths with common random numbers (each household against its no-COVID twin):
    employment: latent Gaussian AR(1) per migrant (phi from CPS flows, abm.employment_phi), employed if
                Phi(z) < CPS rate; stimulus: an out-of-work migrant can send in a stimulus month with
                probability pi; desire: independent monthly draws with probability d_j,t; replacement: each
                lost sender replaced with probability kappa, capped by the reserve (core roles first);
                receipt: u_h,t < s_h(E^c, E^k); transfers: max(E^c + E^k, 1) in months with receipt.
Outputs (aggregates) in outputs/households[-TAG]/:
    persistence_2018.csv  (Step 9 test 4) months with receipt (0-6) in the six months before an ENIGH
        interview (interview months Aug-Nov weighted 10/31/32/27%), 2018 conditions, among households
        receiving at least once: distribution, mean, share in all six. ENIGH 2018: mean 3.9-4.1, 46-50% all six.
    months_2020_vs_2018.csv  (Step 10, ENIGH check) the same distribution in 2020 (COVID run) and 2018, by group.
    who_gains_loses.csv  (Step 10) Apr 2020-Dec 2021 change in months with receipt and in transfers against the
        no-COVID twin, by group (network type, size, strongest tie, main occupation, need features).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

import willing_able as wa
import run_household_model as hm
import run_e1_distribution as e1d

INTERVIEW = {8: 0.10, 9: 0.31, 10: 0.32, 11: 0.27}       # ENIGH interview-month shares (Aug-Nov)
WINDOW = ('2020-04-01', '2021-12-01')


def simulate(pop, inputs, p, kappa, beta, need, pi, stim, seed):
    """One path: (T, H) receipt (bool) and transfers (int). Same seed -> same random numbers (twins)."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    phi = wa.abm.employment_phi()
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need)
    nc, nk = pop['n_core'], pop['n_kin']
    rec, trans = np.zeros((T, H), bool), np.zeros((T, H), int)
    z = streams[0].standard_normal(M)
    hsum = lambda v: np.bincount(hid, weights=v, minlength=H).astype(int)
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        employed = norm.cdf(z) < inputs.rates[t, pop['job_index']]
        checked = streams[2].random(M) < pi * stim[t]
        able = (employed | checked) & pop['eligible']
        wants = streams[3].random(M) < d[t]
        lost = wants & ~able
        cand = lost & (streams[4].random(M) < kappa)
        Wc, Wk = hsum(wants & able & core), hsum(wants & able & ~core)
        Xc, Xk, R = hsum(cand & core), hsum(cand & ~core), hsum(able & ~wants)
        xc = np.minimum(Xc, R)
        xk = np.minimum(Xk, R - xc)
        Ec, Ek = Wc + xc, Wk + xk
        b0, b_any, b_c, b_k, b_k0 = p['b']
        s = wa.expit(b0 + b_any * (Ec > 0) + b_c * np.log1p(Ec) + b_k * np.log1p(Ek)
                     + b_k0 * (nc == 0) * np.log1p(Ek))
        rec[t] = streams[5].random(H) < s
        trans[t] = rec[t] * np.maximum(Ec + Ek, 1)
    return rec, trans


def months_before_interview(rec, dates):
    """(H, 4) months with receipt in the six months before each interview month (Aug..Nov)."""
    idx = {d.month: i for i, d in enumerate(pd.DatetimeIndex(dates))}
    return np.column_stack([rec[[idx[m - k] for k in range(1, 7)]].sum(0) for m in INTERVIEW])


def distribution(months, mask=None):
    """Interview-weighted distribution of months (0-6) among households with at least one month."""
    w = np.array(list(INTERVIEW.values()))
    rows = []
    for j, wj in enumerate(w):
        m = months[:, j] if mask is None else months[mask, j]
        m = m[m > 0]
        rows.append(wj * np.bincount(m, minlength=7)[1:] / max(len(m), 1))
    share = np.sum(rows, axis=0)
    return dict(**{f'share_{k}': share[k - 1] for k in range(1, 7)}, mean=float(share @ np.arange(1, 7)),
                share_all_six=float(share[5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='*', default=None)
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--shocks-tag', default='')
    args = ap.parse_args()
    shocks = wa.OUT / ('shocks' + (f'-{args.shocks_tag}' if args.shocks_tag else ''))
    out = wa.OUT / ('households' + (f'-{args.shocks_tag}' if args.shocks_tag else ''))
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((shocks / 'run_settings.json').read_text())
    fits = pd.read_csv(shocks / 'beta_fit.csv')
    kappas = args.kappas or [float(fits.loc[fits.rmse_fit.idxmin(), 'kappa'])]
    pi, rounds = settings['pi'], settings['stimulus_rounds']

    pop = wa.load_population()
    cal = wa.load_calibration()
    groups = e1d.household_groups(pop)
    need_raw = pop['h']
    groups['need'] = np.select([need_raw.resident_worker.eq(0), need_raw.resident_older.gt(0)],
                               ['No resident worker', 'Older resident'], 'Neither')
    d18 = pd.date_range('2018-01-01', '2018-12-01', freq='MS')
    d20 = pd.date_range('2020-01-01', '2020-12-01', freq='MS')
    in18, in20 = (wa.abm.make_inputs(d, pop['rate_columns']) for d in (d18, d20))
    need20 = hm.need_stock(d20)
    stim20 = wa.stimulus_indicator(d20, rounds)
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need_c, stim_c = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, rounds)
    window = (hm.DATES >= WINDOW[0]) & (hm.DATES <= WINDOW[1])
    zero = np.zeros(len(hm.DATES))

    pers, comp, gain = [], [], []
    for kappa in kappas:
        p = wa.params_for(cal, kappa)
        beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
        acc = {k: [] for k in ('m18', 'm20')}
        dm, dt, rec_cf_any = np.zeros(pop['H']), np.zeros(pop['H']), np.zeros(pop['H'])
        for seed in range(args.seeds):
            r18, _ = simulate(pop, in18, p, kappa, 0.0, None, 0.0, np.zeros(12), 5000 + seed)
            r20, _ = simulate(pop, in20, p, kappa, beta, need20, pi, stim20, 5000 + seed)
            acc['m18'].append(months_before_interview(r18, d18))
            acc['m20'].append(months_before_interview(r20, d20))
            rc, tc = simulate(pop, covid, p, kappa, beta, need_c, pi, stim_c, 7000 + seed)
            rf, tf = simulate(pop, cf, p, kappa, 0.0, None, 0.0, zero, 7000 + seed)
            dm += rc[window].sum(0) - rf[window].sum(0)
            dt += tc[window].sum(0) - tf[window].sum(0)
            rec_cf_any += rf[window].any(0)
        m18, m20 = np.concatenate(acc['m18']), np.concatenate(acc['m20'])
        reps = lambda col: np.tile(groups[col].to_numpy(), args.seeds)
        pers.append(dict(kappa=kappa, beta=beta, **distribution(m18),
                         enigh_2018_mean='3.9-4.1', enigh_2018_share_all_six='0.46-0.50'))
        for col in ['network_type', 'size_bin', 'strongest_tie', 'need']:
            g = reps(col)
            for level in pd.unique(g):
                comp.append(dict(kappa=kappa, group=col, level=str(level), year=2018, **distribution(m18, g == level)))
                comp.append(dict(kappa=kappa, group=col, level=str(level), year=2020, **distribution(m20, g == level)))
        hh = groups.assign(d_months=dm / args.seeds, d_transfers=dt / args.seeds, receives_cf=rec_cf_any / args.seeds)
        for col in ['network_type', 'size_bin', 'strongest_tie', 'occupation', 'need']:
            for level, d in hh.groupby(col, observed=True):
                gain.append(dict(kappa=kappa, beta=beta, group=col, level=str(level), households=len(d),
                                 mean_change_months=float(d.d_months.mean()),
                                 mean_change_transfers=float(d.d_transfers.mean()),
                                 share_losing_months=float((d.d_months < 0).mean()),
                                 share_gaining_months=float((d.d_months > 0).mean())))
        print(f'kappa {kappa}: done', flush=True)
    pd.DataFrame(pers).to_csv(out / 'persistence_2018.csv', index=False)
    pd.DataFrame(comp).to_csv(out / 'months_2020_vs_2018.csv', index=False)
    pd.DataFrame(gain).to_csv(out / 'who_gains_loses.csv', index=False)
    (out / 'run_settings.json').write_text(json.dumps(dict(
        plan='2026-10-30/model_plan.tex, Step 9 test 4 and Step 10', kappas=kappas, seeds=args.seeds,
        stimulus_rounds=rounds, pi=pi, interview_shares=INTERVIEW, window=list(WINDOW),
        employment='latent Gaussian AR(1), phi from CPS (abm.employment_phi), rho = 0'), indent=2))


if __name__ == '__main__':
    main()
