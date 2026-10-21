"""Who lost out? Household distribution of COVID losses with beta = 0 (RQ2, first pass).

Run: python network-compatibility-model/model/run_e1_distribution.py   (~5 min)
     options: --beta 0.06 --kappas 3   (need response with S_t = COVID deaths; output folder
     outputs/e1-distribution-beta<beta>/)

E1a design (beta = 0, rho = 0, pi = 0, base stepping-in rule) for kappa in KAPPAS, with the E0
alphas. Each household is compared with itself: SEEDS paired stochastic runs (COVID vs no-COVID,
common random numbers), window 2020-04..2021-12. Per household, averaged over seeds:
  * expected change in USD received = sum over seeds of COVID USD / sum of no-COVID USD - 1
    (households that never receive in the no-COVID runs are excluded);
  * loss risk = share of seeds in which the household loses more than 25% of its USD;
  * severe interruption risk = share of seeds with a spell of >= 3 consecutive months in which the
    no-COVID twin receives and the COVID household does not.
Expected changes are similar across households (rho = 0: every migrant faces the same occupation-
month odds), so the realized distribution matters more: for each seed (one possible COVID path),
the change of every household against its own no-COVID twin. Common random numbers make this the
loss caused by COVID job losses alone (no luck in sending decisions). realized_bins.csv and
realized_histogram.csv pool household x seed outcomes; realized_inequality.csv averages per seed.
Outputs in outputs/e1-distribution/ are aggregates only (household rows are MMP-derived and
stay in memory): change_bins.csv, change_histogram.csv, groups.csv, inequality.csv.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
import run_e1_job_loss as e1
from run_e0_baseline import household_cells

PROFILE = abm.HERE / 'outputs/e0-baseline/kappa_profile.csv'
KAPPAS = [0.0, 2.0, 4.0]
SEEDS = range(1001, 1201)
WINDOW = ('2020-04-01', '2021-12-01')
BIN_EDGES = [-np.inf, -0.25, -0.10, -0.02, 0.02, 0.10, 0.25, np.inf]
REAL_EDGES = [-np.inf, -0.999, -0.50, -0.25, -0.10, -0.001, 0.001, 0.10, 0.25, np.inf]
REAL_LABELS = ['Lost everything', 'Lost 50–99%', 'Lost 25–50%', 'Lost 10–25%', 'Lost 0–10%', 'No change',
               'Gained 0–10%', 'Gained 10–25%', 'Gained > 25%']
REAL_HIST = np.arange(-1.0, 0.501, 0.05)
BIN_LABELS = ['Lost > 25%', 'Lost 10–25%', 'Lost 2–10%', 'Within ±2%', 'Gained 2–10%', 'Gained 10–25%', 'Gained > 25%']
HIST_EDGES = np.arange(-0.60, 0.401, 0.025)
OCC_ORDER = ['Agriculture', 'Production/construction/transport', 'Sales/personal services',
             'Professional/administrative', 'Other/none']


def longest_spell(missed):
    """missed: (T, H) bool. Longest run of consecutive True per household."""
    run = np.zeros(missed.shape[1], int)
    best = np.zeros(missed.shape[1], int)
    for row in missed:
        run = np.where(row, run + 1, 0)
        best = np.maximum(best, run)
    return best


def household_groups(pop):
    h = pop['h']
    cells = household_cells(h)
    out = cells[['network_type', 'size_bin', 'links']].copy()
    m = pop['m'].assign(hid=pop['hid'], eligible=pop['eligible'])
    job = m.job_group.where(m.job_group.isin(OCC_ORDER[:-1]), 'Other/none')
    dominant = (m.assign(job=job)[m.eligible].groupby('hid').job
                .agg(lambda s: s.value_counts().sort_index().idxmax()))
    out['occupation'] = pd.Series(dominant).reindex(range(pop['H'])).fillna('Other/none').to_numpy()
    out['strongest_tie'] = np.select([h.core_count.gt(0), h.close_kin_count.gt(0)], ['Core', 'Close kin'], 'Distant kin only')
    return out


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def main(beta=0.0, kappas=KAPPAS):
    OUT = abm.HERE / ('outputs/e1-distribution' + (f'-beta{beta:g}' if beta else ''))
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    groups = household_groups(pop)
    covid = abm.make_inputs(e1.DATES, pop['rate_columns'], salience=abm.salience_series(e1.DATES))
    cf = abm.make_inputs(e1.counterfactual_dates(), pop['rate_columns'])
    window = (e1.DATES >= WINDOW[0]) & (e1.DATES <= WINDOW[1])
    profile = pd.read_csv(PROFILE)
    bins, hists, rows, ineq, rbins, rhists, rineq = [], [], [], [], [], [], []
    for kappa in kappas:
        r = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
        params = abm.Params(r.alpha_core, r.alpha_kin, kappa=kappa, beta=beta)
        usd, usd_cf = np.zeros(pop['H']), np.zeros(pop['H'])
        big_loss, severe, half_loss = np.zeros(pop['H']), np.zeros(pop['H']), np.zeros(pop['H'])
        rcount, rhist, rg = np.zeros(len(REAL_LABELS)), np.zeros(len(REAL_HIST) - 1), []
        for seed in SEEDS:
            a = abm.simulate(pop, covid, params, seed, amount_seed=e1.AMOUNT_SEED, keep_households=True)['household'][window]
            b = abm.simulate(pop, cf, params, seed, amount_seed=e1.AMOUNT_SEED, keep_households=True)['household'][window]
            u, ucf = a[:, :, 1].sum(0), b[:, :, 1].sum(0)
            usd += u
            usd_cf += ucf
            big_loss += (ucf > 0) & (u < 0.75 * ucf)
            half_loss += (ucf > 0) & (u < 0.50 * ucf)
            severe += longest_spell((b[:, :, 0] > 0) & (a[:, :, 0] == 0)) >= 3
            rec = ucf > 0
            ch = u[rec] / ucf[rec] - 1
            rcount += np.histogram(ch, REAL_EDGES)[0]
            rhist += np.histogram(ch.clip(REAL_HIST[0], REAL_HIST[-1] - 1e-9), REAL_HIST)[0]
            rg.append((gini(ucf[rec]), gini(u[rec]), float((ch < -0.25).mean()), float((ch <= -0.999).mean()),
                       float(rec.sum())))
        n = len(SEEDS)
        hh = groups.assign(usd=usd / n, usd_cf=usd_cf / n, loss_risk=big_loss / n, severe_risk=severe / n,
                           half_risk=half_loss / n)
        rbins.append(pd.DataFrame({'kappa': kappa, 'bin': REAL_LABELS, 'share': rcount / rcount.sum()}))
        rhists.append(pd.DataFrame({'kappa': kappa, 'bin_left': REAL_HIST[:-1], 'bin_right': REAL_HIST[1:],
                                    'share': rhist / rhist.sum()}))
        rg = np.array(rg)
        rineq.append(dict(kappa=kappa, gini_cf=rg[:, 0].mean(), gini_covid=rg[:, 1].mean(),
                          share_lose_25=rg[:, 2].mean(), share_lose_all=rg[:, 3].mean(),
                          receiving_households=rg[:, 4].mean()))
        hh = hh[hh.usd_cf > 0].copy()
        hh['change'] = hh.usd / hh.usd_cf - 1
        hh['bin'] = pd.cut(hh.change, BIN_EDGES, labels=BIN_LABELS)
        b = hh.groupby('bin', observed=False).agg(households=('change', 'size'), usd_cf=('usd_cf', 'sum'))
        bins.append(pd.DataFrame({'kappa': kappa, 'bin': b.index.astype(str), 'households': b.households.values,
                                  'share_households': b.households.values / len(hh),
                                  'share_cf_usd': b.usd_cf.values / hh.usd_cf.sum()}))
        counts, _ = np.histogram(hh.change.clip(HIST_EDGES[0], HIST_EDGES[-1] - 1e-9), HIST_EDGES)
        hists.append(pd.DataFrame({'kappa': kappa, 'bin_left': HIST_EDGES[:-1], 'bin_right': HIST_EDGES[1:],
                                   'share_households': counts / len(hh)}))
        for g in ['network_type', 'size_bin', 'strongest_tie', 'occupation', 'links']:
            for level, d in hh.groupby(g, observed=True):
                rows.append(dict(kappa=kappa, group=g, level=str(level), households=len(d),
                                 change_usd_weighted=d.usd.sum() / d.usd_cf.sum() - 1,
                                 median_change=d.change.median(),
                                 share_lose_10=float((d.change < -0.10).mean()),
                                 share_lose_25=float((d.change < -0.25).mean()),
                                 share_gain_10=float((d.change > 0.10).mean()),
                                 loss_risk_25=float(d.loss_risk.mean()), loss_risk_50=float(d.half_risk.mean()),
                                 severe_interruption_risk=float(d.severe_risk.mean())))
        loss = (hh.usd_cf - hh.usd).clip(lower=0)
        top = loss.sort_values(ascending=False)
        ineq.append(dict(kappa=kappa, households=len(hh), aggregate_change=hh.usd.sum() / hh.usd_cf.sum() - 1,
                         share_lose_10=float((hh.change < -0.10).mean()), share_lose_25=float((hh.change < -0.25).mean()),
                         share_gain_10=float((hh.change > 0.10).mean()),
                         gini_cf=gini(hh.usd_cf), gini_covid=gini(hh.usd),
                         share_of_losses_top10pct_losers=float(top.iloc[:len(top) // 10].sum() / top.sum()),
                         mean_loss_risk_25=float(hh.loss_risk.mean()), mean_severe_risk=float(hh.severe_risk.mean())))
        print(f'kappa {kappa}: done ({len(hh)} households)')
    tables = dict(change_bins=pd.concat(bins), change_histogram=pd.concat(hists), groups=pd.DataFrame(rows),
                  inequality=pd.DataFrame(ineq), realized_bins=pd.concat(rbins), realized_histogram=pd.concat(rhists),
                  realized_inequality=pd.DataFrame(rineq))
    for k, v in tables.items():
        v.to_csv(OUT / f'{k}.csv', index=False)
    (OUT / 'run_settings.json').write_text(json.dumps(dict(
        experiment='household distribution', beta=beta, rho=0.0, pi=0.0, stepping_in_rule=abm.Params.rule,
        kappas=list(kappas), seeds=[SEEDS.start, SEEDS.stop - 1], window=list(WINDOW), amount_seed=e1.AMOUNT_SEED,
        change='sum over seeds of COVID USD / no-COVID USD - 1, households with no-COVID USD > 0',
        loss_risk='share of seeds with USD < 75% of the paired no-COVID run',
        severe_interruption='share of seeds with >= 3 consecutive missed months (twin receives, COVID household not)',
        occupation='modal job group among the household\'s eligible migrants'), indent=2))
    pd.set_option('display.width', 220)
    print(tables['inequality'].round(3).to_string(index=False))
    print(tables['change_bins'].round(3).to_string(index=False))
    print(tables['realized_inequality'].round(3).to_string(index=False))
    print(tables['realized_bins'].round(3).to_string(index=False))
    g = tables['groups']
    print(g[g.kappa.isin([0, 4])].round(3).to_string(index=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--kappas', type=float, nargs='+', default=KAPPAS)
    a = ap.parse_args()
    main(a.beta, a.kappas)
