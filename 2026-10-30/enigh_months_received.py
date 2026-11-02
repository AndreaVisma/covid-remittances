"""ENIGH: months with receipt among receiving households, 2018 vs 2020, by group (plan Steps 9-10; 30 Oct 2026).

Run (not run yet; ENIGH downloads are untrusted data, so isolated mode):
    python -I 2026-10-30/enigh_months_received.py \
        --ing18 data/external/enigh/unzipped/enigh2018_ns_ingresos_csv/ingresos.csv \
        --ing20 data/external/enigh/unzipped/enigh2020_ns_ingresos_csv/ingresos.csv \
        --fx data/external/fx/fred_DEXMXUS_full.csv
Reuses identification/enigh/enigh_pre_and_monthly.py (household_months, Design, ratio, se, load_inpc, load_fx).
For each household: months with income from abroad (code P041) in the six months before the interview; among
households receiving in at least one month: the distribution of months (1-6), the mean and the share receiving in
all six, with design-based standard errors (Taylor linearization), for all households and by group: resident
worker, resident aged 65+, rural/urban, traditional sending states. Comparable with run_households.py
(months_2020_vs_2018.csv and persistence_2018.csv). Cells with fewer than 30 receiving households are suppressed.
Output (aggregates only): outputs/enigh_months_received.csv.
With --ing22 data/external/enigh/unzipped/enigh2022_ns_ingresos_csv/ingresos.csv (2 Nov 2026): 2018/2020/2022 plus
the share of households receiving, changes vs 2018 (rows year='2020-2018', '2022-2018') and the interview-month
mix, written to outputs/enigh_months_received_2022.csv and enigh_months_received_2022_interview_months.csv.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENIGH = ROOT / 'network-compatibility-model/identification/enigh'
sys.path.insert(0, str(ENIGH))
import enigh_pre_and_monthly as epm          # noqa: E402

DATA = ROOT / 'data'
OUT = HERE / 'outputs'
TRAD = {'01', '06', '10', '11', '14', '16', '18', '24', '32'}   # as identification/enigh/enigh_heterogeneity.py
USE = ['folioviv', 'foliohog', 'ubica_geo', 'tam_loc', 'est_dis', 'upm', 'factor', 'remesas', 'ocupados', 'p65mas']
MIN_N = 30


def load(year):
    d = pd.read_csv(DATA / f'concentradohogar_{year % 100}.csv', usecols=USE,
                    dtype={'folioviv': str, 'foliohog': str, 'ubica_geo': str, 'est_dis': str, 'upm': str},
                    encoding='utf-8-sig')
    d['w'] = d['factor'].astype(float)
    d['psu'] = d['est_dis'] + '_' + d['upm']
    d['state'] = d['ubica_geo'].str.zfill(5).str[:2]
    return d


def groups(d):
    return {'all': np.ones(len(d), bool),
            'no resident worker': (d.ocupados == 0).values, 'resident worker': (d.ocupados > 0).values,
            'resident 65+': (d.p65mas > 0).values, 'no resident 65+': (d.p65mas == 0).values,
            'rural': (d.tam_loc == 4).values, 'urban': (d.tam_loc != 4).values,
            'traditional states': d.state.isin(TRAD).values, 'other states': (~d.state.isin(TRAD)).values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ing18', required=True)
    ap.add_argument('--ing20', required=True)
    ap.add_argument('--ing22', default=None,
                    help='optional; adds 2022, receipt shares, changes vs 2018 and interview months, '
                         'written to enigh_months_received_2022*.csv (the 2018/2020 file is not touched)')
    ap.add_argument('--fx', required=True)
    args = ap.parse_args()
    rp = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s
    inpc, fx = epm.load_inpc(), epm.load_fx(rp(args.fx))
    ext = args.ing22 is not None
    years = [(2018, rp(args.ing18)), (2020, rp(args.ing20))] + ([(2022, rp(args.ing22))] if ext else [])
    rows, ivrows = [], []
    for year, ing in years:
        d = load(year)
        iv, long, _, _ = epm.household_months(ing, year, inpc, fx)
        d = d.merge(iv, on=['folioviv', 'foliohog'], how='left')
        key = d.folioviv + '_' + d.foliohog
        long['key'] = long.folioviv + '_' + long.foliohog
        nmon = long[long.amt > 0].groupby('key').month.nunique()
        months = key.map(nmon).fillna(0).to_numpy()
        recv = (months > 0).astype(float)
        des = epm.Design(d)
        if ext:
            dated = d[d.m1.notna()]
            for m1, g in dated.groupby('m1'):
                ivrows.append(dict(year=year, interview_month=int(m1) + 1, n=len(g),
                                   weighted_share=g.w.sum() / dated.w.sum(),
                                   n_receiving=int((months[d.m1.eq(m1).values] > 0).sum())))
        for name, g in groups(d).items():
            n = int((g & (months > 0)).sum())
            row = dict(year=year, group=name, n_receiving=n)
            if ext:
                r, t = epm.ratio(d, des, recv, np.ones(len(d)), g)[:2]
                row.update(n_households=int(g.sum()), receipt=r, receipt_se=epm.se(t))
            if n >= MIN_N:
                mean, t = epm.ratio(d, des, months, recv, g)[:2]
                row.update(mean=mean, mean_se=epm.se(t))
                for k in range(1, 7):
                    r, t = epm.ratio(d, des, (months == k).astype(float), recv, g)[:2]
                    row.update({f'share_{k}': r, f'share_{k}_se': epm.se(t)})
                row['share_all_six'], row['share_all_six_se'] = row['share_6'], row['share_6_se']
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    if not ext:
        out.to_csv(OUT / 'enigh_months_received.csv', index=False)
        return
    # changes vs 2018 (independent samples: se = sqrt(se_y^2 + se_2018^2)); sample sizes are the later year's
    stats = ['receipt', 'mean'] + [f'share_{k}' for k in range(1, 7)] + ['share_all_six']
    base = out[out.year == 2018].set_index('group')
    ch = []
    for y in (2020, 2022):
        cur = out[out.year == y].set_index('group')
        for gname in cur.index:
            r = dict(year=f'{y}-2018', group=gname, n_receiving=cur.loc[gname, 'n_receiving'],
                     n_households=cur.loc[gname, 'n_households'])
            for s in stats:
                if s in cur.columns and pd.notna(cur.loc[gname, s]) and pd.notna(base.loc[gname, s]):
                    r[s] = cur.loc[gname, s] - base.loc[gname, s]
                    r[s + '_se'] = np.hypot(cur.loc[gname, s + '_se'], base.loc[gname, s + '_se'])
            ch.append(r)
    out = pd.concat([out.astype({'year': str}), pd.DataFrame(ch)], ignore_index=True)
    out.to_csv(OUT / 'enigh_months_received_2022.csv', index=False)
    pd.DataFrame(ivrows).to_csv(OUT / 'enigh_months_received_2022_interview_months.csv', index=False)


if __name__ == '__main__':
    main()
