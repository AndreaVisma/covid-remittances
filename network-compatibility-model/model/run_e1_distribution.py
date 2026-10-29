"""Household grouping and loss-bin helpers for the distribution of COVID losses. Library.

Used by recipient-level/run_household_model.py (household_groups, REAL_*). The migrant-level
distribution runs (outputs/e1-distribution*/) were removed in the 29 Oct 2026 cleanup (git history up to a83b60c).
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
