"""E0: 2019 baseline calibration helpers (model-specification.md §6-§7). Library.

Used by the migrant-level comparison (run_e1_coordination.py) and the household model (target_mask,
household_cells, compare). The stand-alone E0 run under the lost-sending rule (outputs/e0-baseline/) was
removed in the 29 Oct 2026 cleanup (git history up to a83b60c).
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm

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

    def fit(self, kappa, start=START, rule=abm.Params.rule):
        f = least_squares(lambda x: self.residual(abm.Params(*x, kappa=kappa, rule=rule)), start,
                          bounds=([-12, -14], [6, 4]), diff_step=1e-3)
        return abm.Params(*f.x, kappa=kappa, rule=rule), f
