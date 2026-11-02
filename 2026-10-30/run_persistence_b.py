"""Persistence option B (regular and occasional core senders): checks, calibration of lambda, tests (2 Nov 2026).

Run from the repository root:
    python 2026-10-30/run_persistence_b.py checks              lambda = 0 reproduces run_households.simulate; core
                                                               desire draws average to d; exact vs simulated
    python 2026-10-30/run_persistence_b.py calibrate           2018 months-with-receipt distribution, lambda grid
    python 2026-10-30/run_persistence_b.py mig174 --lam X      ever sent over L months, core migrants, 2019
    python 2026-10-30/run_persistence_b.py national --lam X    one shock cell (kappa 1, fitted beta): exact and simulated
    python 2026-10-30/run_persistence_b.py households --lam X  ENIGH-aligned measures, months 2020/2022 vs 2018,
                                                               who gains and loses (kappa 1 and 0.8)
Library: persistence_b.py. Outputs (aggregates only): outputs/persistence-b/.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import willing_able as wa                               # noqa: E402
import run_household_model as hm                        # noqa: E402
import run_households as rh                             # noqa: E402
import run_e1_distribution as e1d                       # noqa: E402
import enigh_aligned as ea                              # noqa: E402
import persistence_b as pb                              # noqa: E402

OUT = wa.OUT / 'persistence-b'
GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
ENIGH18 = dict(mean=4.079, mean_se=0.044, share_1=0.203, share_1_se=0.008, share_all_six=0.496, share_all_six_se=0.011)
D18 = pd.date_range('2018-01-01', '2018-12-01', freq='MS')
D20 = pd.date_range('2020-01-01', '2020-12-01', freq='MS')


class Setup:
    def __init__(self):
        self.pop = pop = wa.load_population()
        self.cal = wa.load_calibration()
        self.fits = pd.read_csv(wa.OUT / 'shocks' / 'beta_fit.csv')
        st = json.loads((wa.OUT / 'shocks' / 'run_settings.json').read_text())
        self.pi, self.rounds = st['pi'], st['stimulus_rounds']
        self.in18 = wa.abm.make_inputs(D18, pop['rate_columns'])
        self.in20 = wa.abm.make_inputs(D20, pop['rate_columns'])
        self.need20, self.stim20 = hm.need_stock(D20), wa.stimulus_indicator(D20, self.rounds)
        self.covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
        self.cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]),
                                     pop['rate_columns'])
        self.need_c, self.stim_c = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, self.rounds)
        self.zero = np.zeros(len(hm.DATES))
        self.groups = e1d.household_groups(pop)
        self.net = self.groups['network_type'].astype(str).to_numpy()

    def beta(self, kappa):
        return float(self.fits.loc[np.isclose(self.fits.kappa, kappa), 'beta'].iloc[0])


def dist_row(months, mask=None):
    return rh.distribution(months, mask)


# %% checks
def checks(S, args):
    pop, p = S.pop, wa.params_for(S.cal, 1.0)
    beta = S.beta(1.0)
    res = {}
    # 1. lambda = 0 reproduces run_households.simulate exactly
    same = True
    for seed in range(5):
        cases = [(S.in18, 0.0, None, 0.0, np.zeros(12)), (S.in20, beta, S.need20, S.pi, S.stim20),
                 (S.covid, beta, S.need_c, S.pi, S.stim_c), (S.cf, 0.0, None, 0.0, S.zero)]
        for kappa in (1.0, 0.8):
            for inp, b, nd, pi, st in cases:
                r0, t0 = rh.simulate(pop, inp, p, kappa, b, nd, pi, st, 5000 + seed)
                r1, t1 = pb.simulate(pop, inp, p, kappa, b, nd, pi, st, 5000 + seed, 0.0)
                same &= bool((r0 == r1).all() and (t0 == t1).all())
    res['lambda0_identical_to_run_households'] = same
    # 2. core desire draws average to d at lambda = 0.8 (COVID run, crisis on); 3. exact vs simulated receipt
    lam, n = 0.8, args.seeds
    want, dexp, recs, trs, clips = [], None, [], [], 0
    for seed in range(n):
        r, t, ex = pb.simulate(pop, S.covid, p, 1.0, beta, S.need_c, S.pi, S.stim_c, 20000 + seed, lam, True)
        want.append(ex['want_core_mean']); dexp = ex['d_core_mean']; clips += ex['n_clip']
        recs.append(r.sum(1)); trs.append(t.sum(1))
    want = np.array(want)
    z_want = (want.mean(0) - dexp) / (want.std(0, ddof=1) / np.sqrt(n))
    a = wa.ability(pop, S.covid, S.pi, S.stim_c)
    d = wa.desire(pop, len(hm.DATES), p['a_c'], p['a_k'], p['eta'], beta, S.need_c, p['g'])
    P, Tr = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, d).pairs(1.0), p['b'])
    recs, trs = np.array(recs), np.array(trs)
    z_rec = (recs.mean(0) - P.sum(1)) / (recs.std(0, ddof=1) / np.sqrt(n))
    z_tr = (trs.mean(0) - Tr.sum(1)) / (trs.std(0, ddof=1) / np.sqrt(n))
    res.update(lambda_check=lam, seeds=n, clipped_cells=clips,
               core_desire_max_abs_z=float(np.abs(z_want).max()),
               core_desire_mean_sim_2019=float(want.mean(0)[:12].mean()), core_desire_d_2019=float(dexp[:12].mean()),
               core_desire_mean_sim_2021_12=float(want.mean(0)[35]), core_desire_d_2021_12=float(dexp[35]),
               receipt_exact_vs_sim_max_abs_z=float(np.abs(z_rec).max()),
               transfers_exact_vs_sim_max_abs_z=float(np.abs(z_tr).max()),
               receipt_exact_mean=float(P.sum(1).mean()), receipt_sim_mean=float(recs.mean()))
    print(json.dumps(res, indent=2))
    (OUT / 'checks.json').write_text(json.dumps(res, indent=2))


# %% calibration on ENIGH 2018
def persistence_2018(S, lam, kappa, seeds, seed0=5000):
    pop, p = S.pop, wa.params_for(S.cal, kappa)
    ms = [rh.months_before_interview(pb.simulate(pop, S.in18, p, kappa, 0.0, None, 0.0, np.zeros(12),
                                                 seed0 + s, lam)[0], D18) for s in range(seeds)]
    return np.concatenate(ms)


def kin_only_share(months, net):
    """Interview-weighted share of receiving households (>= 1 month) that are kin-only."""
    w = np.array(list(rh.INTERVIEW.values()))
    return float(sum(wj * ((months[:, j] > 0) & (net == 'Kin only')).sum() / max((months[:, j] > 0).sum(), 1)
                     for j, wj in enumerate(w)))


def calibrate(S, args):
    rows = []
    grid = args.grid or GRID
    for lam in grid:
        m = persistence_2018(S, lam, 1.0, args.seeds)
        net = np.tile(S.net, args.seeds)
        row = dict(lam=lam, seeds=args.seeds, group='all', **dist_row(m), kin_only_share_of_receivers=kin_only_share(m, net))
        rows.append(row)
        for lev in ['Core only', 'Core + kin', 'Kin only']:
            rows.append(dict(lam=lam, seeds=args.seeds, group=lev, **dist_row(m, net == lev)))
        print(f"lambda {lam}: mean {row['mean']:.3f}, all six {row['share_all_six']:.3f}, one {row['share_1']:.3f}, "
              f"kin-only share of receivers {row['kin_only_share_of_receivers']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    for k, v in ENIGH18.items():
        df[f'enigh_{k}'] = v
    name = 'calibration_2018.csv' if not args.tag else f'calibration_2018_{args.tag}.csv'
    df.to_csv(OUT / name, index=False)
    print(df.round(3).to_string())


# %% mig174
def mig174(S, args):
    p = wa.params_for(S.cal, 0.0)
    cf12 = wa.abm.make_inputs(pd.date_range('2019-01-01', '2019-12-01', freq='MS'), S.pop['rate_columns'])
    L = [1, 3, 6, 12, 24, 60]
    rows = []
    for lam in sorted(set([0.0] + args.lam)):
        r = pb.ever_sends(S.pop, cf12, p, lam, L, args.seeds)
        rows += [dict(lam=lam, months=k, share_sent_any=v) for k, v in r.items()]
    df = pd.DataFrame(rows)
    mmp = {3: 0.70, 6: 0.93, 12: 0.88, 24: 0.89, 60: 0.91}
    df['mmp_mig174'] = df.months.map(mmp)
    df['mmp_bin'] = df.months.map({3: '1-3', 6: '4-6', 12: '7-12', 24: '13-24', 60: '25-60'})
    df.to_csv(OUT / 'mig174_ever_sent.csv', index=False)
    print(df.pivot(index='months', columns='lam', values='share_sent_any').round(3).assign(mmp=lambda x: x.index.map(mmp)))


# %% national check
def national(S, args):
    import run_shocks as rs
    kappa, beta = 1.0, S.beta(1.0)
    sh = rs.Shocks(S.pop, S.cal, S.rounds, S.pi)
    row, _, _ = sh.summary(kappa, beta)
    ref = S.fits.loc[np.isclose(S.fits.kappa, kappa)].iloc[0]
    keys = ['rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'spike_2021_01', 'spike_2021_03',
            'receipt_index_2022', 'tpr_index_2022']
    out = dict(kappa=kappa, beta=beta, exact_max_abs_diff_vs_beta_fit=float(max(abs(row[k] - ref[k]) for k in keys)))
    # simulated national index under option B at lambda (COVID vs no-COVID twins), against the exact index
    p = wa.params_for(S.cal, kappa)
    lam = args.lam[0]
    tc, tf = [], []
    for s in range(args.seeds):
        tc.append(pb.simulate(S.pop, S.covid, p, kappa, beta, S.need_c, S.pi, S.stim_c, 30000 + s, lam)[1].sum(1))
        tf.append(pb.simulate(S.pop, S.cf, p, kappa, 0.0, None, 0.0, S.zero, 30000 + s, lam)[1].sum(1))
    tc, tf = np.array(tc, float), np.array(tf, float)
    qc = np.array([hm.quarterly(pd.Series(x, index=hm.DATES)) for x in tc])
    qf = np.array([hm.quarterly(pd.Series(x, index=hm.DATES)) for x in tf])
    qidx = hm.quarterly(pd.Series(tc[0], index=hm.DATES)).index
    sim = pd.Series(qc.mean(0) / qf.mean(0), index=qidx)
    rc, tcx, rf, tfx = sh.run(kappa, beta)
    exact = hm.quarterly(tcx) / hm.quarterly(tfx)
    # delta-method s.e. of the log ratio of means
    se = np.sqrt(qc.var(0, ddof=1) / qc.mean(0) ** 2 + qf.var(0, ddof=1) / qf.mean(0) ** 2
                 - 2 * np.array([np.cov(qc[:, i], qf[:, i])[0, 1] for i in range(qc.shape[1])]) / (qc.mean(0) * qf.mean(0))) / np.sqrt(args.seeds)
    z = (np.log(sim) - np.log(exact)) / se
    out.update(lam=lam, seeds=args.seeds, sim_index_2020q2=float(sim['2020Q2']), exact_index_2020q2=float(exact['2020Q2']),
               sim_index_2021q4=float(sim['2021Q4']), exact_index_2021q4=float(exact['2021Q4']),
               sim_vs_exact_max_abs_z=float(np.abs(z).max()), sim_vs_exact_max_abs_log_diff=float(np.abs(np.log(sim / exact)).max()))
    pd.DataFrame(dict(quarter=qidx, exact=exact.values, simulated=sim.values, se_log=se, z=z)).to_csv(
        OUT / 'national_check_quarterly.csv', index=False)
    print(json.dumps(out, indent=2))
    (OUT / 'national_check.json').write_text(json.dumps(out, indent=2))


# %% household results
def months_window(rec, dates, year):
    idx = {(d.year, d.month): i for i, d in enumerate(pd.DatetimeIndex(dates))}
    return np.column_stack([rec[[idx[(year, m - k)] for k in range(1, 7)]].sum(0) for m in rh.INTERVIEW])


def households(S, args):
    pop = S.pop
    window = (hm.DATES >= rh.WINDOW[0]) & (hm.DATES <= rh.WINDOW[1])
    enigh = pd.read_csv(ea.ENIGH).query("from_year == 2018 and to_year in (2020, 2022) and measure in ('receipt', 'tpr_counts')").set_index(['measure', 'to_year'])
    aligned, months_rows, gain_rows = [], [], []
    lams = sorted(set([0.0] + args.lam))
    for kappa in args.kappas:
        p, beta = wa.params_for(S.cal, kappa), S.beta(kappa)
        for lam in lams:
            t0 = time.time()
            acc = {y: [] for y in (2020, 2022)}
            m18, m20, m22c, m22 = [], [], [], []
            dm_sum = np.zeros(pop['H'])
            realised = []
            for s in range(args.seeds):
                # ENIGH-aligned (as enigh_aligned.py: seeds 9000+)
                rc, tc = pb.simulate(pop, S.covid, p, kappa, beta, S.need_c, S.pi, S.stim_c, 9000 + s, lam)
                rf, tf = pb.simulate(pop, S.cf, p, kappa, 0.0, None, 0.0, S.zero, 9000 + s, lam)
                for y in acc:
                    acc[y].append((ea.window_measures(rc, tc, hm.DATES, y), ea.window_measures(rf, tf, hm.DATES, y)))
                m22.append(months_window(rc, hm.DATES, 2022)); m22c.append(months_window(rf, hm.DATES, 2022))
                # who gains and loses (paired twins, Apr 2020 - Dec 2021)
                dm = rc[window].sum(0).astype(int) - rf[window].sum(0).astype(int)
                dm_sum += dm
                realised.append(dm)
                # months distribution 2018 and 2020 as run_households (seeds 5000+)
                r18 = pb.simulate(pop, S.in18, p, kappa, 0.0, None, 0.0, np.zeros(12), 5000 + s, lam)[0]
                r20 = pb.simulate(pop, S.in20, p, kappa, beta, S.need20, S.pi, S.stim20, 5000 + s, lam)[0]
                m18.append(rh.months_before_interview(r18, D18)); m20.append(rh.months_before_interview(r20, D20))
            for y, v in acc.items():
                c = np.array([a for a, _ in v]).mean(0); f = np.array([b for _, b in v]).mean(0)
                lr = np.log(c / f)
                aligned.append(dict(kappa=kappa, beta=beta, lam=lam, year=y, seeds=args.seeds,
                                    model_receiving=lr[0], model_tpr=lr[1], model_monthly_receipt=lr[2],
                                    receiving_share_cf=f[0], monthly_receipt_cf=f[2],
                                    enigh_receipt=enigh.loc[('receipt', y), 'excess'], enigh_receipt_se=enigh.loc[('receipt', y), 'excess_se'],
                                    enigh_tpr_counts=enigh.loc[('tpr_counts', y), 'excess'],
                                    enigh_tpr_counts_se=enigh.loc[('tpr_counts', y), 'excess_se']))
            net = np.tile(S.net, args.seeds)
            for name, m in (('2018', m18), ('2020', m20), ('2022', m22), ('2022_no_covid', m22c)):
                m = np.concatenate(m)
                months_rows.append(dict(kappa=kappa, lam=lam, year=name, group='all', **dist_row(m),
                                        receiving_share=float(sum(w * (m[:, j] > 0).mean() for j, w in enumerate(rh.INTERVIEW.values())))))
                for lev in ['Core only', 'Core + kin', 'Kin only']:
                    mk = m[net == lev]
                    months_rows.append(dict(kappa=kappa, lam=lam, year=name, group=lev, **dist_row(m, net == lev),
                                            receiving_share=float(sum(w * (mk[:, j] > 0).mean() for j, w in enumerate(rh.INTERVIEW.values())))))
            realised = np.array(realised)                     # (seeds, H) realised change in months
            mean_dm = dm_sum / args.seeds
            for lev in ['all', 'Core only', 'Core + kin', 'Kin only']:
                sel = np.ones(pop['H'], bool) if lev == 'all' else S.net == lev
                r = realised[:, sel]
                gain_rows.append(dict(kappa=kappa, beta=beta, lam=lam, network_type=lev, households=int(sel.sum()),
                                      seeds=args.seeds, mean_change_months=float(mean_dm[sel].mean()),
                                      expected_share_losing=float((mean_dm[sel] < 0).mean()),
                                      expected_share_gaining=float((mean_dm[sel] > 0).mean()),
                                      realised_share_losing_any=float((r < 0).mean()),
                                      realised_share_losing_3plus=float((r <= -3).mean()),
                                      realised_share_losing_6plus=float((r <= -6).mean()),
                                      realised_share_gaining_any=float((r > 0).mean()),
                                      realised_share_gaining_3plus=float((r >= 3).mean()),
                                      realised_share_unchanged=float((r == 0).mean()),
                                      realised_sd_change=float(r.std())))
            print(f'kappa {kappa} lambda {lam}: {time.time() - t0:.0f}s', flush=True)
    tag = f'_{args.tag}' if args.tag else ''
    pd.DataFrame(aligned).to_csv(OUT / f'enigh_aligned{tag}.csv', index=False)
    pd.DataFrame(months_rows).to_csv(OUT / f'months_by_year{tag}.csv', index=False)
    pd.DataFrame(gain_rows).to_csv(OUT / f'who_gains_loses{tag}.csv', index=False)
    print(pd.DataFrame(aligned).round(3).to_string())
    print(pd.DataFrame(gain_rows).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['checks', 'calibrate', 'mig174', 'national', 'households'])
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--lam', type=float, nargs='*', default=[0.0])
    ap.add_argument('--grid', type=float, nargs='*', default=None)
    ap.add_argument('--kappas', type=float, nargs='*', default=[1.0, 0.8])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    S = Setup()
    dict(checks=checks, calibrate=calibrate, mig174=mig174, national=national, households=households)[args.step](S, args)
    print(f'{args.step}: {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
