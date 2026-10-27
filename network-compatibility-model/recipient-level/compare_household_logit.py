"""Recipient-level receipt logit vs the migrant-level model, 2019 MMP (27 Oct 2026).

Run: python network-compatibility-model/recipient-level/compare_household_logit.py   (~2 min)

Question: is household receipt better described by a household-level logit than by the
migrant-level model's independent senders? The migrant-level model implies
    P(h receives in t) = 1 - prod_j (1 - e_j,t * theta_j)        (kappa = 0)
a household function with a fixed shape: every extra migrant adds an independent chance.
The household logit keeps migrants only for the employment shock and models receipt directly:
    P(h receives in t) = E[ expit(a + g_c(A_c) + g_k(A_k) + eta' need_h) ]
with A_c, A_k the numbers of able core and kin migrants (Poisson-binomial under the 2019 CPS
employment of each migrant, rho = 0, exact) and g a linear or log(1 + n) term.

Both families are fitted the same way: maximum likelihood on the same MMP households
(e0.target_mask, receipt = remus), 2019 CPS employment, S = 0, no benefits, kappa = 0.
Specifications:
  migrant level   M1: alpha_core, alpha_kin                     M2: M1 + eta_worker, eta_older
                  M1f, M2f: with a household floor, P = 1 - (1 - expit(floor)) prod_j (1 - e_j theta_j)
                  (receipt from sources other than the linked migrants; without it, households whose
                  migrants are all 'Not working' have P = 0, and the 21 of them that receive dominate
                  the likelihood)
  household logit H0: linear able core and kin counts           H1: log(1 + count) terms
                  H2: H1 + eta_worker, eta_older                H3: H2 + any able core migrant
References (not refitted): M1-e0 (E0 alphas, kappa = 0) and M2-chi2 (outputs/e0-need, lost_sending,
kappa = 0), calibrated on cell means rather than by likelihood.
Need features are abm.need_features (centred, incomplete rosters at the mean) in both families.

Comparison: log-likelihood, AIC, BIC; 5-fold cross-validated log loss with folds by community (also on
households with at least one eligible migrant);
chi-square on the 12 need cells of run_e0_need; weighted RMSE (pp) on the E0 validation panels;
and the implied 2020-21 job-loss response (household receipt index, COVID / same month of 2019,
kappa = 0, no need response), since the household logit responds differently to losing a migrant.
Outputs (aggregates only): outputs/fits.csv, comparison.csv, validation.csv, panel_rmse.csv,
shock_response.csv, run_settings.json.

Variant --not-working-as-unknown (outputs/not-working-unknown/): migrants whose MMP job group is
'Not working' (1,417 of 7,999; abm treats them as never able to send) get the all-occupation CPS rate,
like migrants with unknown occupation. Motivation: households whose core migrants are all 'Not
working' still receive in 43% of cases, so the code likely describes the last trip, not current
ability to send. The E0 reference calibrations are skipped (fitted under the original coding).
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))
import abm
import run_e0_baseline as e0
import run_e0_need as e0n

OUT = HERE / 'outputs'
FOLDS = 5
SEED = 7
PMF_TOL = 1e-12
SHOCK_DATES = pd.date_range('2020-01-01', '2021-12-01', freq='MS')
HOUSEHOLD_SPECS = {
    'H0': ['core', 'kin'],
    'H1': ['log_core', 'log_kin'],
    'H2': ['log_core', 'log_kin', 'no_worker', 'any_older'],
    'H3': ['log_core', 'log_kin', 'no_worker', 'any_older', 'any_core'],
}
MIGRANT_SPECS = {'M1': ['alpha_core', 'alpha_kin'],
                 'M2': ['alpha_core', 'alpha_kin', 'eta_worker', 'eta_older'],
                 'M1f': ['alpha_core', 'alpha_kin', 'floor'],
                 'M2f': ['alpha_core', 'alpha_kin', 'eta_worker', 'eta_older', 'floor']}


# %% Able-migrant counts (exact, rho = 0)
def poisson_binomial(P):
    """(T, n) success probabilities -> (T, n + 1) pmf of the number of successes."""
    out = np.ones((P.shape[0], 1))
    for j in range(P.shape[1]):
        q = P[:, j:j + 1]
        out = np.pad(out * (1 - q), ((0, 0), (0, 1))) + np.pad(out * q, ((0, 0), (1, 0)))
    return out


def able_pmfs(pop, inputs):
    """Per household: (pmf of able core migrants, pmf of able kin migrants), each (T, n + 1)."""
    e = abm.able_probability(pop, inputs, abm.Params(0.0, 0.0))
    order = np.argsort(pop['hid'], kind='stable')
    bounds = np.searchsorted(pop['hid'][order], np.arange(pop['H'] + 1))
    out = []
    for h in range(pop['H']):
        idx = order[bounds[h]:bounds[h + 1]]
        core = pop['core'][idx]
        out.append((poisson_binomial(e[:, idx[core]]), poisson_binomial(e[:, idx[~core]])))
    return out


def design(a, b, need, terms):
    cols = {'core': a, 'kin': b, 'log_core': np.log1p(a), 'log_kin': np.log1p(b),
            'any_core': (a > 0).astype(float), 'no_worker': need[:, 0], 'any_older': need[:, 1]}
    return np.column_stack([np.ones(len(a))] + [cols[t] for t in terms])


# %% Models
class HouseholdLogit:
    """Receipt = expit(design) averaged over the joint pmf of able core and kin counts."""

    def __init__(self, pmfs, need, terms):
        rows = []
        for h, (pc, pk) in enumerate(pmfs):
            W = np.einsum('ta,tb->ab', pc, pk) / len(pc)
            a, b = np.nonzero(W > PMF_TOL)
            rows.append((np.full(len(a), h), a, b, W[a, b]))
        self.hh, a, b, self.w = map(np.concatenate, zip(*rows))
        self.H, self.terms = len(pmfs), terms
        self.X = design(a.astype(float), b.astype(float), need[self.hh], terms)
        self.names = ['intercept'] + terms
        self.start = np.r_[-1.0, np.zeros(len(terms))]

    def prob(self, x):
        return np.bincount(self.hh, self.w * expit(self.X @ x), minlength=self.H)

    def negloglik(self, x, y, mask):
        s = expit(self.X @ x)
        P = np.clip(np.bincount(self.hh, self.w * s, minlength=self.H), 1e-12, 1 - 1e-12)
        r = np.where(mask, y / P - (1 - y) / (1 - P), 0.0)
        ll = np.sum(np.where(mask, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))
        return -ll, -(self.X.T @ (r[self.hh] * self.w * s * (1 - s)))

    def fit(self, y, mask, start=None):
        res = minimize(self.negloglik, self.start if start is None else start, args=(y, mask),
                       jac=True, method='BFGS', options=dict(gtol=1e-7, maxiter=2000))
        return res.x

    def shock_receipt(self, x, pmfs, need):
        """(T, H) receipt probabilities, month by month."""
        out = np.zeros((len(pmfs[0][0]), len(pmfs)))
        for h, (pc, pk) in enumerate(pmfs):
            A, B = pc.shape[1], pk.shape[1]
            aa, bb = np.meshgrid(np.arange(A, dtype=float), np.arange(B, dtype=float), indexing='ij')
            X = design(aa.ravel(), bb.ravel(), np.repeat(need[h:h + 1], A * B, axis=0), self.terms)
            S = expit(X @ x).reshape(A, B)
            out[:, h] = np.einsum('ta,ab,tb->t', pc, S, pk)
        return out


class MigrantModel:
    """The ABM's exact receipt at kappa = 0: P = 1 - prod_j (1 - e_j theta_j), independent senders.
    Closed form instead of abm.expected_receipt (1.4 s per call); main() checks they agree."""

    def __init__(self, pop, inputs, names):
        self.pop, self.inputs, self.names = pop, inputs, names
        self.floor = names[-1] == 'floor'
        self.e = {id(inputs): abm.able_probability(pop, inputs, abm.Params(0.0, 0.0))}

    @staticmethod
    def params(x):
        x = list(x) + [0.0] * (4 - len(x))
        return abm.Params(x[0], x[1], eta_worker=x[2], eta_older=x[3])

    def receipt(self, x, inputs):
        """(T, H) monthly receipt probabilities. With a floor (last parameter, logit scale), the
        household also receives with probability expit(floor) from sources other than its linked
        migrants: P = 1 - (1 - expit(floor)) * prod_j (1 - e_j theta_j)."""
        if id(inputs) not in self.e:
            self.e[id(inputs)] = abm.able_probability(self.pop, inputs, abm.Params(0.0, 0.0))
        e, hid, H = self.e[id(inputs)], self.pop['hid'], self.pop['H']
        p = self.params(x[:-1] if self.floor else x)
        theta = expit(np.where(self.pop['core'], p.alpha_core, p.alpha_kin) + abm.need_shift(self.pop, p)[hid])
        log_none = np.log1p(-np.clip(e * theta, 0, 1 - 1e-12))
        none = np.exp(np.stack([np.bincount(hid, w, minlength=H) for w in log_none]))
        return 1 - (1 - expit(x[-1])) * none if self.floor else 1 - none

    def prob(self, x, inputs=None):
        return self.receipt(x, inputs or self.inputs).mean(axis=0)

    def negloglik(self, x, y, mask):
        P = np.clip(self.prob(x), 1e-12, 1 - 1e-12)
        return -np.sum(np.where(mask, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))

    def fit(self, y, mask, start=None):
        res = minimize(self.negloglik, start, args=(y, mask), method='L-BFGS-B',
                       options=dict(ftol=1e-12, gtol=1e-8, maxiter=500))
        return res.x

    def shock_receipt(self, x, inputs):
        return self.receipt(x, inputs)


def numerical_se(f, x, step=1e-3):
    k = len(x)
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * step, np.eye(k)[j] * step
            Hm[i, j] = Hm[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * step ** 2)
    try:
        return np.sqrt(np.diag(np.linalg.inv(Hm)))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


# %% Diagnostics
def chi2_need_cells(cells_src, p):
    known = cells_src.workers.ne('Unknown').to_numpy()
    g = cells_src[known].assign(p=p[known]).groupby(e0n.FIT_BY)
    m_net = cells_src.groupby('network_type').received.mean()
    t = g.agg(mmp=('received', 'mean'), model=('p', 'mean'), n=('received', 'count')).reset_index()
    m = m_net.reindex(t.network_type).to_numpy()
    return float(np.sum(((t.model - t.mmp) / np.sqrt(m * (1 - m) / t.n)) ** 2))


def validation(cells_src, cells_sim, p, model):
    out = []
    for panel, by in e0n.PANELS:
        t = e0.compare(cells_src, cells_sim, p, by)
        if 'network_type' not in by:
            t = t.assign(network_type='All')
        t = t.rename(columns={by[-1]: 'category'}) if by[-1] != 'network_type' else t.assign(category='All')
        out.append(t.assign(model=model, panel=panel))
    return pd.concat(out, ignore_index=True)


def community_folds(h, ok):
    rng = np.random.default_rng(SEED)
    comm = (h.country.astype(str) + '-' + h.commun.astype(str)).to_numpy()
    ids = rng.permutation(np.unique(comm[ok]))
    fold_of = dict(zip(ids, np.arange(len(ids)) % FOLDS))
    return np.array([fold_of.get(c, -1) for c in comm])


def receipt_index(P_covid, P_cf, dates):
    t = pd.DataFrame({'date': dates, 'receipt_covid': P_covid.mean(1), 'receipt_cf': P_cf.mean(1)})
    t['index'] = t.receipt_covid / t.receipt_cf
    return t


# %% Main
def load_population(not_working_as_unknown=False):
    pop = abm.load_population()
    if not_working_as_unknown:
        m = pop['m']
        nw = m.job_group.eq('Not working').to_numpy()
        adult = m.age_years.ge(18).fillna(False).to_numpy()
        pop['job_index'] = np.where(nw, pop['rate_columns'].index(abm.ALL_OCC), pop['job_index'])
        pop['eligible'] = pop['eligible'] | (nw & adult)
        pop['has_eligible_core'] = np.bincount(pop['hid'], weights=pop['core'] & pop['eligible'],
                                               minlength=pop['H']) > 0
    return pop


def main(not_working_as_unknown=False):
    out = OUT / 'not-working-unknown' if not_working_as_unknown else OUT
    out.mkdir(parents=True, exist_ok=True)
    pop = load_population(not_working_as_unknown)
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    cells_sim = e0n.cells(h).loc[ok]
    cells_src = cells_sim.assign(received=y[ok])
    need = pop['need_x']
    inputs = abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns'])
    pmfs = able_pmfs(pop, inputs)
    folds = community_folds(h, ok)
    n_obs = int(ok.sum())

    models = {name: HouseholdLogit(pmfs, need, terms) for name, terms in HOUSEHOLD_SPECS.items()}
    models.update({name: MigrantModel(pop, inputs, names) for name, names in MIGRANT_SPECS.items()})
    e0_row = pd.read_csv(abm.HERE / 'outputs/e0-baseline/kappa_profile.csv').query('kappa == 0').iloc[0]
    need_row = pd.read_csv(abm.HERE / 'outputs/e0-need/fits.csv').query(
        "rule == 'lost_sending' and kappa == 0 and model == 'need'").iloc[0]
    starts = {'M1': np.array([e0_row.alpha_core, e0_row.alpha_kin]),
              'M2': need_row[['alpha_core', 'alpha_kin', 'eta_worker', 'eta_older']].to_numpy(float)}
    starts.update(M1f=np.r_[starts['M1'], -3.0], M2f=np.r_[starts['M2'], -3.0])
    has_eligible = np.bincount(pop['hid'], weights=pop['eligible'], minlength=pop['H']) > 0
    sub = ok & has_eligible
    references = {'M1-e0': ('M1', starts['M1']), 'M2-chi2': ('M2', starts['M2'])}
    for name, x in [(n, starts[n]) for n in ('M1', 'M2')]:
        gap = np.abs(models[name].prob(x) - abm.expected_receipt(pop, inputs, MigrantModel.params(x)).mean(0)).max()
        assert gap < 1e-10, f'closed form differs from abm.expected_receipt for {name}: {gap}'

    fits, comp, valid, estimates = [], [], [], {}

    def evaluate(label, model, x, cv_loss, k, method):
        p = model.prob(x)
        nll = model.negloglik(x, y, ok)
        nll = nll[0] if isinstance(nll, tuple) else nll
        P = np.clip(p, 1e-12, 1 - 1e-12)
        ll_sub = float(np.sum((y * np.log(P) + (1 - y) * np.log(1 - P))[sub]))
        comp.append(dict(model=label, family='household logit' if label.startswith('H') else 'migrant level',
                         estimation=method, k=k, loglik=-nll, aic=2 * k + 2 * nll, bic=k * np.log(n_obs) + 2 * nll,
                         cv_logloss=cv_loss[0], loglik_eligible=ll_sub, cv_logloss_eligible=cv_loss[1],
                         chi2_need_cells=chi2_need_cells(cells_src, p[ok]),
                         mean_receipt_model=p[ok].mean(), mean_receipt_mmp=y[ok].mean()))
        valid.append(validation(cells_src, cells_sim, p[ok], label))

    for label, model in models.items():
        start = starts.get(label)
        x = model.fit(y, ok, start)
        f = (lambda z, m=model: (lambda v: v[0] if isinstance(v, tuple) else v)(m.negloglik(z, y, ok)))
        se = numerical_se(f, x)
        estimates[label] = x
        for nm, b, s in zip(model.names, x, se):
            fits.append(dict(model=label, parameter=nm, estimate=b, se=s))
        loss = np.full(pop['H'], np.nan)
        for k in range(FOLDS):
            test, train = ok & (folds == k), ok & (folds != k)
            xk = model.fit(y, train, x)
            P = np.clip(model.prob(xk), 1e-12, 1 - 1e-12)
            loss[test] = -(y * np.log(P) + (1 - y) * np.log(1 - P))[test]
        evaluate(label, model, x, (float(np.nanmean(loss[ok])), float(np.nanmean(loss[sub]))), len(x),
                 'maximum likelihood')
        print(f'{label}: ' + ', '.join(f'{n} {b:.3f} ({s:.3f})' for n, b, s in zip(model.names, x, se)), flush=True)

    for label, (base, x) in ({} if not_working_as_unknown else references).items():
        evaluate(label, models[base], x, (np.nan, np.nan), len(x), 'cell means (E0 / e0-need)')
        estimates[label] = x

    # Implied 2020-21 job-loss response (kappa = 0, no need response).
    covid = abm.make_inputs(SHOCK_DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([d.replace(year=2019) for d in SHOCK_DATES]), pop['rate_columns'])
    pm_covid, pm_cf = able_pmfs(pop, covid), able_pmfs(pop, cf)
    shock = []
    for label, model in models.items():
        x = estimates[label]
        if isinstance(model, HouseholdLogit):
            Pc, Pf = model.shock_receipt(x, pm_covid, need), model.shock_receipt(x, pm_cf, need)
        else:
            Pc, Pf = model.shock_receipt(x, covid), model.shock_receipt(x, cf)
        shock.append(receipt_index(Pc, Pf, SHOCK_DATES).assign(model=label))
    shock = pd.concat(shock, ignore_index=True)

    fits, comp = pd.DataFrame(fits), pd.DataFrame(comp)
    valid = pd.concat(valid, ignore_index=True)
    valid = valid[valid.category.ne('Unknown')]
    rel = valid[valid.reliable]
    rmse = (rel.assign(sq=rel.error_pp ** 2).groupby(['model', 'panel'])
            .apply(lambda g: np.sqrt(np.average(g.sq, weights=g.source_n)), include_groups=False)
            .rename('weighted_rmse_pp').reset_index())
    fits.to_csv(out / 'fits.csv', index=False)
    comp.to_csv(out / 'comparison.csv', index=False)
    valid.to_csv(out / 'validation.csv', index=False)
    rmse.to_csv(out / 'panel_rmse.csv', index=False)
    shock.to_csv(out / 'shock_response.csv', index=False)
    settings = dict(
        experiment='Recipient-level receipt logit vs migrant-level model, 2019 MMP',
        not_working_as_unknown=not_working_as_unknown, eligible_migrants=int(pop['eligible'].sum()),
        households=n_obs, receiving=int(y[ok].sum()), migrants=int(pop['M']),
        households_with_eligible_migrant=int(sub.sum()),
        eligible_note='loglik_eligible / cv_logloss_eligible: households with at least one migrant who can ever send '
                      '(not all migrants coded Not working); without a floor the migrant model gives the others P = 0',
        employment='2019 CPS rates by occupation, exact Poisson-binomial able counts, rho = 0',
        receipt='MMP remus vs mean monthly receipt probability (E0 mapping)', kappa=0,
        household_specs=HOUSEHOLD_SPECS, migrant_specs=MIGRANT_SPECS,
        cv=dict(folds=FOLDS, grouping='community (country, commun)', seed=SEED),
        shock=dict(dates=[str(SHOCK_DATES[0].date()), str(SHOCK_DATES[-1].date())],
                   counterfactual='2019 CPS rates of the same calendar month', response='job loss only'))
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2))

    pd.set_option('display.width', 220)
    print(comp.round(3).to_string(index=False))
    print(rmse.pivot(index='panel', columns='model', values='weighted_rmse_pp').round(2))
    q = shock.assign(q=shock.date.dt.to_period('Q')).groupby(['model', 'q'])[['receipt_covid', 'receipt_cf']].mean()
    print((q.receipt_covid / q.receipt_cf).unstack('model').round(3))


if __name__ == '__main__':
    main(not_working_as_unknown='--not-working-as-unknown' in sys.argv)
