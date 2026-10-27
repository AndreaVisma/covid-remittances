"""E0 with household need: does need at home shift the 2019 sending probability?

Run: python network-compatibility-model/model/run_e0_need.py   (~30 min)

Sending logit with a household need shift (abm.need_shift):
    logit theta_j = alpha_type + beta * S_t + eta_worker * no_worker_h + eta_dep * dependency_h
no_worker_h: no resident worker; dependency_h: (resident children + older residents) / residents;
both centred on the population mean, so alpha stays the propensity of an average household.

Calibration as in E0 (cell means, not household likelihood: a household likelihood also chases the
known size-gradient misfit of core networks and pulls alpha_core down). Households with a known
roster are partitioned into network type x worker present x dependency bin (18 cells). Minimum
chi-square: residual = (model - MMP receipt) / sqrt(m (1 - m) / n), with m the MMP receipt of the
cell's network type. Model receipt is the exact mean monthly probability under 2019 CPS employment
(rho = 0, S = 0, no benefits), the E0 mapping. For each rule and kappa:
  * 'alpha': alpha_core, alpha_kin only (nested model);
  * 'need':  alpha_core, alpha_kin, eta_worker, eta_dep; covariance (J'J)^-1; test of eta = 0 by the
    drop in chi-square (2 df).
The E0 alphas (least squares on receipt by network type) are reported as 'e0' for reference.
Validation: model vs MMP receipt by network type, workers, dependency, children, older residents,
network size, link type and survey period (aggregates only).
Outputs: outputs/e0-need/fits.csv, validation.csv, panel_rmse.csv, run_settings.json.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
import run_e0_baseline as e0

OUT = abm.HERE / 'outputs/e0-need'
RULES = {'coordination': (abm.HERE / 'outputs/e1-coordination/e0_kappa_profile.csv', [0.0, 0.25, 0.5, 0.7, 1.0]),
         'lost_sending': (abm.HERE / 'outputs/e0-baseline/kappa_profile.csv', [0.0, 2.0, 4.0])}
DEP_BINS = [-0.01, 0.0, 0.5, 1.01]
DEP_LABELS = ['No dependents', 'Up to half', 'Over half']  # not 'None': read back as NaN
FIT_BY = ['network_type', 'workers', 'dependency']
PANELS = [('network_type', ['network_type']), ('workers', ['network_type', 'workers']),
          ('dependency', ['network_type', 'dependency']), ('children', ['network_type', 'children']),
          ('older', ['network_type', 'older']), ('size', ['network_type', 'size_bin']),
          ('links', ['links']), ('survey_period', ['survey_period'])]
NAMES = ['alpha_core', 'alpha_kin', 'eta_worker', 'eta_dep']


def cells(h):
    out = e0.household_cells(h)
    dep = (h.resident_child + h.resident_older) / h.resident.where(h.resident > 0)
    out['dependency'] = pd.cut(dep, DEP_BINS, labels=DEP_LABELS).astype(object).fillna('Unknown')
    out['workers'] = np.where(h.resident_worker.isna() | out.dependency.eq('Unknown'), 'Unknown', out['workers'])
    return out


class MinChi2:
    def __init__(self, pop, inputs, cells_src, ok):
        self.pop, self.inputs, self.ok = pop, inputs, ok
        known = cells_src[cells_src.workers.ne('Unknown')]
        g = known.groupby(FIT_BY).received.agg(['mean', 'count'])
        m_net = cells_src.groupby('network_type').received.mean()
        self.target = g.reset_index()
        self.goal = g['mean'].to_numpy()
        m = m_net.reindex(self.target.network_type).to_numpy()
        self.sd = np.sqrt(m * (1 - m) / g['count'].to_numpy())
        keys = pd.MultiIndex.from_frame(cells_src[FIT_BY])
        self.cell_of = pd.Index(g.index).get_indexer(keys)            # -1: unknown roster, not fitted

    def receipt(self, params):
        return abm.expected_receipt(self.pop, self.inputs, params).mean(axis=0)

    @staticmethod
    def params(x, kappa, rule):
        x = list(x) + [0.0] * (4 - len(x))
        return abm.Params(x[0], x[1], kappa=kappa, rule=rule, eta_worker=x[2], eta_dep=x[3])

    def residual(self, x, kappa, rule):
        p = self.receipt(self.params(x, kappa, rule))[self.ok]
        live = self.cell_of >= 0
        model = np.bincount(self.cell_of[live], weights=p[live], minlength=len(self.goal)) / \
            np.bincount(self.cell_of[live], minlength=len(self.goal))
        return (model - self.goal) / self.sd

    def fit(self, start, kappa, rule):
        n = len(start)
        lo, hi = [-12, -14, -6, -6][:n], [6, 4, 6, 6][:n]
        return least_squares(self.residual, start, args=(kappa, rule), bounds=(lo, hi), diff_step=1e-3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.loc[ok, 'remus'].eq(1).to_numpy(float)
    cells_sim = cells(h).loc[ok]
    cells_src = cells_sim.assign(received=y)
    inputs = abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns'])
    cal = MinChi2(pop, inputs, cells_src, ok)
    n_cells = len(cal.goal)

    fits, valid = [], []
    for rule, (profile_path, kappas) in RULES.items():
        profile = pd.read_csv(profile_path)
        for kappa in kappas:
            row = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
            x_e0 = np.array([row.alpha_core, row.alpha_kin])
            f_a = cal.fit(x_e0, kappa, rule)
            f_n = cal.fit(np.r_[f_a.x, 0.0, 0.0], kappa, rule)
            cov = np.linalg.inv(f_n.jac.T @ f_n.jac)
            se = np.sqrt(np.diag(cov))
            chi_a, chi_n = 2 * f_a.cost, 2 * f_n.cost
            for name, x in {'e0': x_e0, 'alpha': f_a.x, 'need': f_n.x}.items():
                chi = float(np.sum(cal.residual(x, kappa, rule) ** 2))
                k = len(x)
                rec = dict(rule=rule, kappa=kappa, model=name, **dict(zip(NAMES, list(x) + [0.0] * (4 - k))),
                           chi2=chi, df=n_cells - k, fit_p=chi2.sf(chi, n_cells - k))
                if name == 'need':
                    rec.update({f'se_{nm}': s for nm, s in zip(NAMES, se)}, delta_chi2=chi_a - chi_n,
                               eta_test_p=chi2.sf(chi_a - chi_n, 2), corr_eta=cov[2, 3] / (se[2] * se[3]))
                fits.append(rec)
                p_sim = cal.receipt(cal.params(x, kappa, rule))[ok]
                for panel, by in PANELS:
                    t = e0.compare(cells_src, cells_sim, p_sim, by)
                    if 'network_type' not in by:
                        t = t.assign(network_type='All')
                    t = t.rename(columns={by[-1]: 'category'}) if by[-1] != 'network_type' else t.assign(category='All')
                    valid.append(t.assign(rule=rule, kappa=kappa, model=name, panel=panel))
            print(f'{rule} kappa {kappa}: eta_worker {f_n.x[2]:.3f} ({se[2]:.3f}), eta_dep {f_n.x[3]:.3f} '
                  f'({se[3]:.3f}), chi2 {chi_a:.1f} -> {chi_n:.1f} on {n_cells} cells, '
                  f'p(eta = 0) {chi2.sf(chi_a - chi_n, 2):.3g}', flush=True)

    fits, valid = pd.DataFrame(fits), pd.concat(valid, ignore_index=True)
    valid = valid[valid.category.ne('Unknown')]
    rel = valid[valid.reliable]
    rmse = (rel.assign(sq=rel.error_pp ** 2)
            .groupby(['rule', 'kappa', 'model', 'panel'])
            .apply(lambda g: np.sqrt(np.average(g.sq, weights=g.source_n)), include_groups=False)
            .rename('weighted_rmse_pp').reset_index())
    fits.to_csv(OUT / 'fits.csv', index=False)
    valid.to_csv(OUT / 'validation.csv', index=False)
    rmse.to_csv(OUT / 'panel_rmse.csv', index=False)
    raw = np.column_stack([h.resident_worker.eq(0), (h.resident_child + h.resident_older) / h.resident.where(h.resident > 0)])
    settings = dict(
        experiment='E0 with household need (2019, minimum chi-square on MMP receipt cells)',
        logit='alpha_type + beta*S_t + eta_worker*no_worker_h + eta_dep*dependency_h (need centred)',
        need_features=dict(no_worker='no resident worker (resident_worker == 0)',
                           dependency='(resident_child + resident_older) / resident',
                           missing='incomplete roster or no residents: mean (0 after centring), not in fit cells',
                           means=dict(zip(abm.NEED_FEATURES, map(float, np.nanmean(raw, axis=0))))),
        fit_cells=dict(by=FIT_BY, dependency_bins=dict(zip(DEP_LABELS, ['0', '(0, 0.5]', '(0.5, 1]'])),
                       n_cells=n_cells, households=int((cal.cell_of >= 0).sum()),
                       sd='sqrt(m (1 - m) / n), m = MMP receipt of the network type'),
        receipt_mapping='MMP remus vs exact mean monthly receipt probability, 2019 CPS, rho = 0, S = 0',
        rules={r: dict(profile=str(p.relative_to(abm.HERE)), kappas=k) for r, (p, k) in RULES.items()},
        households=int(ok.sum()), receiving=int(y.sum()))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))

    pd.set_option('display.width', 220)
    print(cal.target.round(3).to_string(index=False))
    print(fits.round(3).to_string(index=False))
    print(rmse.pivot_table(index=['rule', 'kappa', 'panel'], columns='model', values='weighted_rmse_pp').round(2))


if __name__ == '__main__':
    main()
