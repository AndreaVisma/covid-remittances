"""Need acting on margins other than the household-level shift: experiments (29 Oct 2026).

Run: python network-compatibility-model/recipient-level/need_margins.py   (~15 min)
     --quick: skip the ENIGH-window simulation (heterogeneity table without the 6-month column)
     --het-only: reuse outputs/need-margins/fits.csv, redo only the heterogeneity table

Experiments. The base model (run_household_model.py) adopted 6a with series E and lambda = 1 on
29 Oct 2026 (user decision); before that its need term was the household shift H with the raw S_t.
Same population, 2019 calibration, kappa (replacement share), stimulus (pi = 1 in Jan and Mar 2021,
0 in Apr 2020), Banxico target and fit window. Each experiment puts need N*_t on one margin:

  H   household shift (the base form, for reference)  logit + beta N*_t                 (all households)
  6a  more senders per receiving household            theta_j,t = expit(alpha_j + beta_s N*_t), all migrants
  6b  same, kin only                                  theta_j,t = expit(alpha_kin + beta_s N*_t), kin only
      (option 6 leaves receipt unchanged; counts = sum_h P_h,t x E[senders | receive]_h,t)
  7a  dormant ties: kin-only households               logit + beta_k N*_t [no core migrant]
  7b  dormant ties: vulnerable households             logit + beta_v N*_t [no resident worker or older resident]
  8   persistence: N*_t = sum_k lambda^k N_t-k        (lambda = 0: flow; lambda = 1: cumulative stock)

Need series N_t (data-prep/outputs/salience_inputs_monthly.csv, aggregate public series; 0 before
Mar 2020 and in the no-COVID run):
  S  reported COVID deaths (the base S_t)
  E  excess deaths (World Mortality Dataset, trend baseline, negatives set to 0)
  L  Mexican employment loss vs same month of 2019 (OECD/ETOE, negatives set to 0)
  R  placebo: a constant flow from Mar 2020 (lambda = 0: a step; lambda = 1: a linear ramp)
Every stock N*_t is scaled to peak 1 over 2020-03..2021-12, so beta is the logit shift at the peak.

Count mappings. Base: transfers per receiving household constant except for need-activated senders,
    index_t = sum_h P_c,h,t m_h,t / sum_h P_f,h,t,  m_h,t = r_h,t(e_c, theta(N*)) / r_h,t(e_c, theta(0)),
    r = E[able senders | any sends] = sum_j q_j / (1 - prod_j (1 - q_j)), q_j = e_j,t theta_j,t
    (migrant-level M1 thetas, as run_household_model.transfers_per_receiver); m = 1 for H, 7, and N = 0.
Lower bound: index_t = sum_h P_c r(e_c, theta(N*)) / sum_h P_f r(e_f, theta(0)) (as map.csv *_lower_bound).
One free parameter per option (beta), fitted by minimising the quarterly log RMSE 2020Q2..2021Q4 at
fixed kappa (0, 0.7, 0.8); lambda on a grid (option 8).

Check: 6a with E and lambda = 1 reproduces the base model's outputs/household-model/map.csv
(assertion, kappa x beta cells, base and lower-bound mappings).

Heterogeneity predictions (kappa = 0.8, each margin at its best series and lambda): change in expected
receipt and transfers, 2021 vs the no-COVID run (2019 conditions), by group; and the ENIGH 2020
window: share of households receiving at least once in the six months before the interview month,
interview Aug-Nov 2020 (equal weights), from 100 paired realized paths (employment as abm.simulate).
Outputs (aggregates only) in outputs/need-margins/.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))
sys.path.insert(0, str(HERE))
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1
import run_e1_distribution as e1d
import compare_household_logit as hl
import run_household_model as hm

OUT = HERE / 'outputs/need-margins'
INPUTS = HERE.parent / 'data-prep/outputs/salience_inputs_monthly.csv'
DATES, FIT_Q, HOLD_Q = hm.DATES, hm.FIT_Q, hm.HOLD_Q
KAPPAS = [0.0, 0.7, 0.8]
LAMBDAS = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]
SERIES = {'S': 'covid_deaths', 'E': 'excess_deaths_trend', 'L': 'employment_loss', 'R': None}
SERIES_LABEL = {'S': 'reported COVID deaths', 'E': 'excess deaths', 'L': 'Mexican employment loss',
                'R': 'placebo: constant flow from Mar 2020 (lambda = 1: linear ramp)'}
MARGINS = {
    'H': 'household shift, all households (base form)',
    '6a': 'more senders: all migrants (base mapping)',
    '6b': 'more senders: kin only (base mapping)',
    '6a_lb': 'more senders: all migrants (lower-bound mapping)',
    '6b_lb': 'more senders: kin only (lower-bound mapping)',
    '7a': 'dormant ties: kin-only households',
    '7b': 'dormant ties: no resident worker or older resident',
}
BETA_MAX = 8.0
HET_KAPPA = 0.8
ENIGH_INTERVIEW = pd.date_range('2020-08-01', '2020-11-01', freq='MS')
ENIGH_DATES = pd.date_range('2020-02-01', '2020-10-01', freq='MS')
ENIGH_SEEDS = range(3001, 3101)
STATE_NAMES = {1: 'Aguascalientes', 2: 'Baja California', 9: 'Ciudad de Mexico', 10: 'Durango',
               11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco', 15: 'Mexico', 16: 'Michoacan',
               17: 'Morelos', 18: 'Nayarit', 19: 'Nuevo Leon', 20: 'Oaxaca', 21: 'Puebla', 22: 'Queretaro',
               24: 'San Luis Potosi', 25: 'Sinaloa', 27: 'Tabasco', 29: 'Tlaxcala', 30: 'Veracruz',
               31: 'Yucatan', 32: 'Zacatecas'}


# %% Need series
def need_flow(code, dates=DATES):
    if code == 'R':     # placebo: no information on need, only on timing
        d = pd.Series(1.0, index=pd.date_range('2019-01-01', '2022-12-01', freq='MS'))
    else:
        d = pd.read_csv(INPUTS, index_col='date', parse_dates=True)[SERIES[code]].clip(lower=0)
    d[d.index < '2020-03-01'] = 0.0
    return d.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy(float)


def need_stock(code, lam):
    """N*_t on DATES, scaled to peak 1 over 2020-03..2021-12."""
    n = need_flow(code)
    out = np.zeros_like(n)
    for t in range(len(n)):
        out[t] = n[t] + (lam * out[t - 1] if t else 0.0)
    peak = out[(DATES >= '2020-03-01') & (DATES <= '2021-12-01')].max()
    return out / peak


# %% Model pieces (copies of run_household_model with a household-specific shift)
def receipt(cells, x, kappa, shift):
    """(T, H) receipt; shift (T, H) logit shift (hm.Cells.receipt has a common beta S_t)."""
    s = expit((cells.X @ x)[None, :] + shift[:, cells.hh])
    s_full = expit((cells.X_full @ x)[None, :] + shift)
    return (cells.household_sum(cells.W * s * (1 - kappa * cells.trigger))
            + kappa * s_full * cells.household_sum(cells.W * cells.trigger))


class Senders:
    """E[able senders | any sends] per household and month, migrant-level M1 thetas."""

    def __init__(self, pop, inputs):
        fits = pd.read_csv(HERE / 'outputs/not-working-unknown/fits.csv').query("model == 'M1'").set_index('parameter')
        self.alpha = np.where(pop['core'], fits.estimate['alpha_core'], fits.estimate['alpha_kin'])
        e = abm.able_probability(pop, inputs, abm.Params(0.0, 0.0))
        self.e = e + (1 - e) * hm.stimulus_pi(inputs.dates)[:, None] * pop['eligible'][None, :]
        self.hid, self.H, self.kin = pop['hid'], pop['H'], ~pop['core']

    def hsum(self, Z):
        return np.stack([np.bincount(self.hid, weights=z, minlength=self.H) for z in Z])

    def r(self, shift_t=None, kin_only=False):
        """shift_t: (T,) logit shift of theta (all migrants or kin only)."""
        a = np.broadcast_to(self.alpha, self.e.shape)
        if shift_t is not None:
            a = a + shift_t[:, None] * (self.kin if kin_only else np.ones_like(self.kin))[None, :]
        q = self.e * expit(a)
        p_any = -np.expm1(self.hsum(np.log1p(-q)))
        return np.divide(self.hsum(q), p_any, out=np.ones_like(p_any), where=p_any > 1e-12)


class Experiment:
    def __init__(self):
        pop = self.pop = hl.load_population(not_working_as_unknown=True)
        h = pop['h']
        ok = e0.target_mask(h)
        y = h.remus.eq(1).to_numpy(float)
        base = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
        self.x = {}
        start = None
        for kappa in sorted(set(KAPPAS)):
            start = self.x[kappa] = hm.fit_2019(base, y, ok, kappa, start)[0]
        self.covid = abm.make_inputs(DATES, pop['rate_columns'], salience=abm.salience_series(DATES))
        self.cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in DATES]), pop['rate_columns'])
        self.cc, self.cfc = hm.Cells(pop, self.covid), hm.Cells(pop, self.cf)
        self.snd_c, self.snd_f = Senders(pop, self.covid), Senders(pop, self.cf)
        self.r_c0, self.r_f0 = self.snd_c.r(), self.snd_f.r()
        self.Pf = {k: receipt(self.cfc, self.x[k], k, np.zeros((len(DATES), pop['H']))) for k in KAPPAS}
        self.Pc0 = {k: receipt(self.cc, self.x[k], k, np.zeros((len(DATES), pop['H']))) for k in KAPPAS}
        known = h.resident.gt(0) & h[['resident_worker', 'resident_older']].notna().all(axis=1)
        self.no_core = (hm.eligible_counts(pop)[0] == 0).astype(float)
        self.no_worker = (known & h.resident_worker.eq(0)).to_numpy()
        self.older = (known & h.resident_older.gt(0)).to_numpy()
        self.vulnerable = (self.no_worker | self.older).astype(float)
        self.known = known.to_numpy()
        target, info = hm.banxico_target()
        self.target = target
        bx_m, trend_m, _ = e1.banxico()
        self.spike_bx = hm.spikes((bx_m / trend_m).reindex(DATES))

    # household-level paths for one margin
    def paths(self, margin, kappa, beta, N):
        H = self.pop['H']
        shift = np.zeros((len(DATES), H))
        if margin == 'H':
            shift = shift + beta * N[:, None]
        elif margin == '7a':
            shift = beta * N[:, None] * self.no_core[None, :]
        elif margin == '7b':
            shift = beta * N[:, None] * self.vulnerable[None, :]
        Pc = self.Pc0[kappa] if not shift.any() else receipt(self.cc, self.x[kappa], kappa, shift)
        Pf = self.Pf[kappa]
        if margin.startswith('6') and beta:
            r_c = self.snd_c.r(beta * N, kin_only=margin.startswith('6b'))
        else:
            r_c = self.r_c0
        Tc = Pc * r_c / self.r_c0       # base mapping
        return dict(Pc=Pc, Pf=Pf, Tc=Tc, Tf=Pf, Lc=Pc * r_c, Lf=Pf * self.r_f0)

    def index(self, margin, kappa, beta, N):
        p = self.paths(margin, kappa, beta, N)
        lb = margin.endswith('_lb')
        c, f = (p['Lc'], p['Lf']) if lb else (p['Tc'], p['Tf'])
        mc, mf = pd.Series(c.sum(1), index=DATES), pd.Series(f.sum(1), index=DATES)
        return hm.quarterly(mc) / hm.quarterly(mf), mc / mf

    def summary(self, margin, kappa, beta, N):
        q, m = self.index(margin, kappa, beta, N)
        sp = hm.spikes(m)
        return dict(rmse_fit=hm.log_rmse(q, self.target, FIT_Q), rmse_holdout_2022=hm.log_rmse(q, self.target, HOLD_Q),
                    index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                    mean_index_2021=float(q[[s for s in FIT_Q if s.startswith('2021')]].mean()),
                    mean_index_2022=float(q[HOLD_Q].mean()),
                    spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                    spike_rmse=float(np.sqrt(np.mean([(sp[k] - self.spike_bx[k]) ** 2 for k in sp])))), q

    def fit(self, margin, kappa, N):
        f = lambda b: self.summary(margin, kappa, b, N)[0]['rmse_fit']
        res = minimize_scalar(f, bounds=(0.0, BETA_MAX), method='bounded', options=dict(xatol=1e-3))
        b = float(res.x) if res.fun < f(0.0) else 0.0
        return b

    # verification against map.csv
    def verify(self):
        grid = pd.read_csv(HERE / 'outputs/household-model/map.csv')
        N = need_stock('E', 1.0)
        worst = 0.0
        for kappa in KAPPAS:
            for beta in [0.0, 0.3, 0.65]:
                row = grid[np.isclose(grid.kappa, kappa) & np.isclose(grid.beta, beta)].iloc[0]
                s, _ = self.summary('6a', kappa, beta, N)
                s_lb, _ = self.summary('6a_lb', kappa, beta, N)
                pairs = [(s['rmse_fit'], row.rmse_log_quarterly), (s['index_2020q2'], row.index_2020q2),
                         (s['rmse_holdout_2022'], row.rmse_log_quarterly_holdout_2022),
                         (s['spike_2021_01'], row['spike_2021-01']), (s['spike_2021_03'], row['spike_2021-03']),
                         (s['mean_index_2021'], row.mean_index_2021),
                         (s_lb['rmse_fit'], row.rmse_log_quarterly_lower_bound),
                         (s_lb['index_2020q2'], row.index_2020q2_lower_bound),
                         (s_lb['spike_2021_01'], row['spike_2021-01_lower_bound'])]
                worst = max(worst, max(abs(a - b) for a, b in pairs))
        assert worst < 1e-6, f'does not reproduce map.csv: max abs difference {worst}'
        return worst


# %% Heterogeneity
def groups(ex):
    pop, h = ex.pop, ex.pop['h']
    g = e1d.household_groups(pop)[['network_type', 'size_bin']].copy()
    yn = lambda v: np.where(~ex.known, 'Unknown roster', np.where(v, 'Yes', 'No'))
    g['no_resident_worker'] = yn(ex.no_worker)
    g['any_older_resident'] = yn(ex.older)
    g['state'] = h.origin_state_code.map(lambda c: STATE_NAMES.get(int(c), str(int(c))) if pd.notna(c) else 'Unknown')
    return g


def simulate_receipt(pop, inputs, x, kappa, seed, shift):
    """Copy of hm.simulate_receipt with a (T, H) logit shift."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    phi = abm.employment_phi()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    pi_t = hm.stimulus_pi(inputs.dates)
    z = streams[0].standard_normal(M)
    n_core, n_kin = hm.eligible_counts(pop)
    feats = hm.design_features(pop)
    out = np.zeros((T, H), bool)
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        employed = norm.cdf(z) < inputs.rates[t, pop['job_index']]
        checked = streams[3].random(M) < pi_t[t]
        able = (employed | checked) & pop['eligible']
        ac = np.bincount(hid, weights=able & core, minlength=H)
        ak = np.bincount(hid, weights=able & ~core, minlength=H)
        p = hm.household_receipt(x, kappa, 'share', ac, ak, n_core, n_kin, feats, shift=shift[t])
        out[t] = streams[2].random(H) < p
    return out


def enigh_any6(ex, margin, beta, N_full):
    """(H,) share of (seed, interview month) with receipt in the 6 months before, COVID and no-COVID."""
    pop, kappa = ex.pop, HET_KAPPA
    sel = DATES.isin(ENIGH_DATES)
    N = N_full[sel]
    shift = np.zeros((len(ENIGH_DATES), pop['H']))
    if margin == 'H':
        shift += beta * N[:, None]
    elif margin == '7a':
        shift = beta * N[:, None] * ex.no_core[None, :]
    elif margin == '7b':
        shift = beta * N[:, None] * ex.vulnerable[None, :]
    covid = abm.make_inputs(ENIGH_DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([d.replace(year=2019) for d in ENIGH_DATES]), pop['rate_columns'])
    out_c, out_f = np.zeros(pop['H']), np.zeros(pop['H'])
    for seed in ENIGH_SEEDS:
        a = simulate_receipt(pop, covid, ex.x[kappa], kappa, seed, shift)
        b = simulate_receipt(pop, cf, ex.x[kappa], kappa, seed, np.zeros_like(shift))
        for m in ENIGH_INTERVIEW:
            w = (ENIGH_DATES >= m - pd.DateOffset(months=6)) & (ENIGH_DATES < m)
            out_c += a[w].any(0)
            out_f += b[w].any(0)
    n = len(ENIGH_SEEDS) * len(ENIGH_INTERVIEW)
    return out_c / n, out_f / n


def heterogeneity(ex, chosen, run_enigh):
    g = groups(ex)
    y21 = DATES.year == 2021
    rows = []
    for margin, (code, lam, beta) in chosen.items():
        N = need_stock(code, lam) if code else np.zeros(len(DATES))
        p = ex.paths(margin, HET_KAPPA, beta, N)
        cols = {k: v[y21].sum(0) for k, v in p.items()}
        if run_enigh:
            e6c, e6f = enigh_any6(ex, margin.replace('_lb', ''), beta, N)
            cols.update(E6c=e6c, E6f=e6f)
        d = g.assign(**cols)
        for grp in ['all', 'no_resident_worker', 'any_older_resident', 'network_type', 'size_bin', 'state']:
            parts = [('All', d)] if grp == 'all' else d.groupby(grp, observed=True)
            for level, s in parts:
                r = dict(margin=margin, series=code or '-', lam=lam, beta=beta, kappa=HET_KAPPA, group=grp,
                         level=str(level), households=len(s),
                         receipt_2019=s.Pf.sum() / 12 / len(s),
                         share_of_transfers_2019=s.Tf.sum() / d.Tf.sum(),
                         change_receipt_2021=s.Pc.sum() / s.Pf.sum() - 1,
                         change_transfers_2021=s.Tc.sum() / s.Tf.sum() - 1,
                         change_transfers_2021_lower_bound=s.Lc.sum() / s.Lf.sum() - 1)
                if run_enigh:
                    r.update(enigh_6m_share_nocovid=s.E6f.mean(), enigh_6m_share_covid=s.E6c.mean(),
                             change_enigh_6m_2020=s.E6c.mean() / s.E6f.mean() - 1)
                rows.append(r)
        print(f'heterogeneity {margin}: done', flush=True)
    return pd.DataFrame(rows)


# %% Main
def main(run_enigh=True, het_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ex = Experiment()
    worst = ex.verify()
    print(f'verification vs map.csv: max abs difference {worst:.2e}', flush=True)
    bx = ex.target
    rows, qpaths = [], []
    base = {}
    for kappa in KAPPAS if not het_only else []:
        s, q = ex.summary('H', kappa, 0.0, np.zeros(len(DATES)))
        base[kappa] = s
        rows.append(dict(margin='none', series='-', lam=np.nan, kappa=kappa, beta=0.0, **s, gap_closed_2021q4=0.0))
        qpaths.append(pd.DataFrame(dict(margin='none', series='-', lam=np.nan, kappa=kappa, quarter=q.index,
                                        index_model=q.values, index_banxico=bx[q.index].values)))
    for margin in MARGINS if not het_only else []:
        for code in SERIES:
            for lam in LAMBDAS:
                N = need_stock(code, lam)
                for kappa in KAPPAS:
                    if kappa == 0.0 and lam not in (0.0, 1.0):
                        continue
                    beta = ex.fit(margin, kappa, N)
                    s, q = ex.summary(margin, kappa, beta, N)
                    gap = np.log(bx['2021Q4']) - np.log(base[kappa]['index_2021q4'])
                    closed = (np.log(s['index_2021q4']) - np.log(base[kappa]['index_2021q4'])) / gap
                    rows.append(dict(margin=margin, series=code, lam=lam, kappa=kappa, beta=beta, **s,
                                     gap_closed_2021q4=closed, beta_at_bound=beta > BETA_MAX - 0.01))
                    qpaths.append(pd.DataFrame(dict(margin=margin, series=code, lam=lam, kappa=kappa, quarter=q.index,
                                                    index_model=q.values, index_banxico=bx[q.index].values)))
            print(f'{margin} {code}: done', flush=True)
    if het_only:
        res = pd.read_csv(OUT / 'fits.csv')
    else:
        res = pd.DataFrame(rows)
        res.to_csv(OUT / 'fits.csv', index=False)
        pd.concat(qpaths, ignore_index=True).to_csv(OUT / 'quarterly_paths.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['margin', 'series', 'lam', 'kappa', 'beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2',
            'index_2021q4', 'mean_index_2022', 'spike_2021_01', 'spike_2021_03', 'gap_closed_2021q4']
    print(res[cols].round(3).to_string(index=False))

    # best (need series, lambda) per margin at HET_KAPPA, for the heterogeneity table (placebo R excluded)
    k8 = res[np.isclose(res.kappa, HET_KAPPA) & res.margin.ne('none') & res.series.ne('R')]
    best = k8.loc[k8.groupby('margin').rmse_fit.idxmin()]
    best.to_csv(OUT / 'best_by_margin.csv', index=False)
    chosen = {'none': (None, 0.0, 0.0)}
    chosen.update({r.margin: (r.series, r.lam, r.beta) for r in best.itertuples() if not r.margin.endswith('_lb')})
    het = heterogeneity(ex, chosen, run_enigh)
    het.to_csv(OUT / 'heterogeneity.csv', index=False)
    print(het.round(3).to_string(index=False))

    stocks = pd.DataFrame({'date': DATES, **{f'{c}_lambda{l:g}': need_stock(c, l) for c in SERIES for l in LAMBDAS}})
    stocks.to_csv(OUT / 'need_series.csv', index=False)
    settings = dict(
        experiment='Need on other margins (options 6, 7, 8); experiments, not the base model',
        kappas=KAPPAS, lambdas=LAMBDAS, series=SERIES_LABEL,
        scaling='N*_t = sum_k lambda^k N_t-k, scaled to peak 1 over 2020-03..2021-12; 0 before 2020-03',
        margins=MARGINS, beta_bounds=[0.0, BETA_MAX], base_need_term='beta S_t switched off (beta = 0)',
        criterion='quarterly log RMSE 2020Q2..2021Q4 (beta fitted); 2022 held out',
        stimulus=hm.PI_STIMULUS, spike_banxico=ex.spike_bx,
        verification=dict(reference='outputs/household-model/map.csv', max_abs_difference=worst),
        heterogeneity=dict(kappa=HET_KAPPA, choice='best need series (S, E, L; not the placebo R) and lambda by fit RMSE per margin',
                           receipt='2021 expected receipt months, COVID / no-COVID run',
                           transfers='base mapping (and lower bound)',
                           enigh=None if not run_enigh else dict(
                               interview_months=[str(d.date()) for d in ENIGH_INTERVIEW],
                               statistic='received at least once in the 6 months before the interview month',
                               seeds=[ENIGH_SEEDS.start, ENIGH_SEEDS.stop - 1],
                               note='option 6 does not change receipt: its ENIGH prediction equals none')))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=float))


if __name__ == '__main__':
    main(run_enigh='--quick' not in sys.argv, het_only='--het-only' in sys.argv)
