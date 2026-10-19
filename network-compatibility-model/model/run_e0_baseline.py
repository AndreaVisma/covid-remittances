"""E0: 2019 baseline calibration (model-specification.md §6-§7).

Run: python network-compatibility-model/model/run_e0_baseline.py

For each kappa (kin stepping-in when the household's eligible core migrants
cannot send), fits alpha_core and alpha_kin to MMP household receipt rates by
network type (core only / kin only / core + kin), using the exact expected
monthly receipt probability under 2019 CPS employment (rho = 0, no salience,
no benefits). Then:
  * kappa profile: best 2019 fit at each kappa (how much 2019 constrains kappa);
  * untargeted validation: receipt by network size and household composition;
  * stochastic check: simulated receipt equals the analytic probability;
  * what kappa changes in 2019 (transfers, triggered households, USD) and the
    Banxico count scale per kappa;
  * ENIGH-style check: six-month average USD among receiving households.

Assumption carried from the old ABM: MMP 'remus' (household receives US
remittances) is compared with the monthly receipt probability.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm

OUT = abm.HERE / 'outputs/e0-baseline'
NETWORKS = ['Core only', 'Kin only', 'Core + kin']
SIZE_BINS = [1, 2, 3, 5, 10, 10_000]
SIZE_LABELS = ['1', '2', '3-4', '5-9', '10+']
PERIOD_BINS = [2010, 2014, 2017, 2020]
PERIOD_LABELS = ['2010-13', '2014-16', '2017-19']
KAPPAS = [0.0, 1.0, 2.0, 3.0, 4.0]                 # reported grid
KAPPA_PROFILE = np.arange(0, 8.01, 0.5)            # fit-quality profile
SEEDS = [11, 12, 13, 14, 15]
MIN_SOURCE_N = 30
START = np.array([0.0, -4.0])


def household_cells(frame):
    out = pd.DataFrame(index=frame.index)
    out['network_type'] = abm.network_type(frame)
    out['size_bin'] = pd.cut(frame.network_size, SIZE_BINS, right=False, labels=SIZE_LABELS)
    out['links'] = np.select([frame.core_count.gt(0), frame.close_kin_count.gt(0) & frame.distant_link.gt(0),
                              frame.close_kin_count.gt(0)],
                             ['Has core', 'Close + distant kin', 'Close kin only'], 'Distant kin only')
    out['survey_period'] = pd.cut(frame.surveyyr, PERIOD_BINS, right=False, labels=PERIOD_LABELS)
    out['workers'] = np.where(frame.resident_worker.eq(0), 'No worker', 'Worker(s)')
    out['children'] = np.where(frame.resident_child.gt(0), 'Children', 'No children')
    out['older'] = np.where(frame.resident_older.gt(0), 'Older residents', 'No older residents')
    return out


def target_mask(h):
    """Households used for fitting and validation: complete records, known receipt.
    The population contains every connected MMP household once, so the model
    is compared with the same households it simulates."""
    return (h.recorded_fields_complete.astype(bool) & h.remus.isin([1, 2])).to_numpy()


def load_targets(h, mask):
    cells = household_cells(h).loc[mask]
    cells['received'] = h.loc[mask, 'remus'].eq(1).astype(float)
    return cells


def compare(cells_src, cells_sim, p_sim, by):
    src = cells_src.groupby(by, observed=True).received.agg(source_n='count', mmp_receipt='mean')
    sim = pd.DataFrame({'p': p_sim}, index=cells_sim.index).join(cells_sim)
    sim = sim.groupby(by, observed=True).p.agg(synthetic_n='count', model_receipt='mean')
    out = src.join(sim, how='outer').reset_index()
    out['error_pp'] = 100 * (out.model_receipt - out.mmp_receipt)
    out['reliable'] = out.source_n.ge(MIN_SOURCE_N)
    return out


class Calibrator:
    def __init__(self, pop, inputs, cells_src, cells_sim, sim_ok):
        self.pop, self.inputs, self.sim_ok = pop, inputs, sim_ok
        target = cells_src.groupby('network_type').received.agg(['mean', 'count']).loc[NETWORKS]
        self.goal = target['mean'].to_numpy()
        self.weight = np.sqrt(target['count'] / target['count'].sum()).to_numpy()
        self.masks = [(cells_sim.network_type == n).to_numpy() for n in NETWORKS]

    def receipt(self, params):
        return abm.expected_receipt(self.pop, self.inputs, params).mean(axis=0)

    def residual(self, params):
        p = self.receipt(params)[self.sim_ok]
        return self.weight * (np.array([p[mk].mean() for mk in self.masks]) - self.goal)

    def fit(self, kappa, start=START):
        f = least_squares(lambda x: self.residual(abm.Params(*x, kappa=kappa)), start,
                          bounds=([-12, -14], [6, 4]), diff_step=1e-3)
        return abm.Params(*f.x, kappa=kappa), f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    h = pop['h']
    sim_ok = target_mask(h)
    cells_sim = household_cells(h).loc[sim_ok]
    cells_src = load_targets(h, sim_ok)
    inputs = abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns'])
    cal = Calibrator(pop, inputs, cells_src, cells_sim, sim_ok)

    # --- kappa profile -------------------------------------------------------
    profile, start = [], START
    for kappa in KAPPA_PROFILE:
        params, f = cal.fit(kappa, start)
        start = np.array([params.alpha_core, params.alpha_kin])
        profile.append(dict(kappa=kappa, alpha_core=params.alpha_core, alpha_kin=params.alpha_kin,
                            weighted_rmse_pp=100 * np.sqrt(np.mean(f.fun ** 2)), success=f.success))
    profile = pd.DataFrame(profile)
    profile.to_csv(OUT / 'kappa_profile.csv', index=False)

    # --- fit, validation and stochastic runs on the reported grid ------------
    inputs2 = abm.make_inputs(abm.baseline_2019_dates(repeats=2), pop['rate_columns'])
    bx = pd.read_parquet(abm.BANXICO)
    banxico_monthly = bx.loc[pd.to_datetime(bx.date).dt.year.eq(2019)].transactions.mean()
    fits, sizes, comps, runs, enighs = [], [], [], [], []
    for kappa in KAPPAS:
        row = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
        params = abm.Params(row.alpha_core, row.alpha_kin, kappa=kappa)
        p_hh = cal.receipt(params)
        p_sim = p_hh[sim_ok]
        fits.append(compare(cells_src, cells_sim, p_sim, ['network_type']).assign(kappa=kappa))
        sizes.append(compare(cells_src, cells_sim, p_sim, ['network_type', 'size_bin']).assign(kappa=kappa))
        for c in ['links', 'survey_period']:
            comps.append(compare(cells_src, cells_sim, p_sim, [c])
                         .rename(columns={c: 'category'}).assign(network_type='All', panel=c, kappa=kappa))
        for c in ['workers', 'children', 'older']:
            comps.append(compare(cells_src, cells_sim, p_sim, ['network_type', c])
                         .rename(columns={c: 'category'}).assign(panel=c, kappa=kappa))
        for seed in SEEDS:
            r = abm.simulate(pop, inputs2, params, seed, amount_seed=7, keep_households=True)
            s, hh = r['summary'], r['household']
            receiving = hh[:, :, 0] > 0
            runs.append(dict(kappa=kappa, seed=seed, alpha_core=params.alpha_core, alpha_kin=params.alpha_kin,
                             analytic_receipt_rate=p_hh.mean(), simulated_receipt_rate=s.receipt_rate.mean(),
                             transfers_per_month=s.transfers.mean(), core_transfers=s.core_transfers.mean(),
                             kin_transfers=s.kin_transfers.mean(), triggered_households=s.triggered_households.mean(),
                             kin_transfers_triggered=s.kin_transfers_triggered.mean(),
                             multi_transfer_rate=s.multi_transfer_rate.mean(),
                             usd_per_receiving_household_month=np.nanmean(np.where(receiving, hh[:, :, 1], np.nan)),
                             mean_transfer_usd=s.usd.sum() / s.transfers.sum(),
                             employed_share=s.employed_share.mean(),
                             banxico_scale=banxico_monthly / s.transfers.mean()))
            window, any6 = hh[-6:, :, 1].mean(axis=0), receiving[-6:].any(axis=0)
            enighs.append(dict(kappa=kappa, seed=seed, share_receiving_6m=any6.mean(),
                               median_usd_month=np.median(window[any6]), mean_usd_month=window[any6].mean()))
    fit_table, size_table, comp_table = (pd.concat(x, ignore_index=True) for x in [fits, sizes, comps])
    kap = pd.DataFrame(runs).groupby('kappa').mean().drop(columns='seed').reset_index()
    enigh = pd.DataFrame(enighs).groupby('kappa').mean().drop(columns='seed').reset_index()
    fit_table.to_csv(OUT / 'fit_network_type.csv', index=False)
    size_table.to_csv(OUT / 'validation_network_size.csv', index=False)
    comp_table.to_csv(OUT / 'validation_composition.csv', index=False)
    kap.to_csv(OUT / 'kappa_2019.csv', index=False)
    enigh.to_csv(OUT / 'enigh_style_check.csv', index=False)

    amounts = json.loads(abm.AMOUNTS.read_text())
    settings = dict(kappa_rule='logit theta_kin = alpha_kin + beta*S_t + kappa*1[eligible core, none able]',
                    fitted=profile.loc[profile.kappa.isin(KAPPAS)].to_dict('records'),
                    phi=abm.employment_phi(), kappas=KAPPAS, seeds=SEEDS,
                    households=pop['H'], migrants=pop['M'], core_migrants=int(pop['core'].sum()),
                    eligible_migrants=int(pop['eligible'].sum()),
                    households_with_eligible_core=int(pop['has_eligible_core'].sum()),
                    fit_households=int(sim_ok.sum()), source_households=len(cells_src),
                    banxico_2019_monthly_transactions=float(banxico_monthly),
                    enigh_2018_recipient_monthly_usd=dict(
                        median=amounts['enigh_2018_validation']['recipient_monthly_usd_median'],
                        mean=amounts['enigh_2018_validation']['recipient_monthly_usd_mean']),
                    receipt_mapping='MMP remus compared with monthly receipt probability',
                    cps_population=json.loads(abm.CPS_SETTINGS.read_text())['population'])
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))

    pd.set_option('display.width', 220)
    print(profile.round(3).to_string(index=False))
    print(fit_table.round(3).to_string(index=False))
    print(size_table.loc[size_table.kappa.isin([0, 4])].round(3).to_string(index=False))
    print(comp_table.loc[comp_table.kappa.eq(0) & (comp_table.panel.isin(['links', 'survey_period'])
                                                   | comp_table.error_pp.abs().gt(5))].round(3).to_string(index=False))
    print(kap.round(3).to_string(index=False))
    print(enigh.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
