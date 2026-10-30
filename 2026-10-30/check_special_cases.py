"""Checks of the willing-and-able library before any result is used (30 Oct 2026; not run yet).

Run: python 2026-10-30/check_special_cases.py      (minutes; no outputs written, asserts and prints)
1. Probabilities: every household's (E^c, E^k) distribution sums to 1 in every month, for kappa 0, 0.5, 1;
   E[senders] equals sum_j a_j d_j when kappa = 0.
2. Nesting: with desire = 1 for every migrant (a_c = a_k = 30, no need, no crisis) and kappa = 0, effective
   senders are the able migrants, so receipt must equal the current household model (run_household_model.Cells,
   kappa = 0) with the same receipt coefficients and its need coefficients set to 0. Max abs difference < 1e-10.
3. Stepping-in: receipt and transfers are non-decreasing in kappa, and households with a single migrant are
   unaffected by kappa (no reserve can step in).
4. Exact vs simulated: for kappa = 0.8 and illustrative coefficients, the mean of simulated receipt over
   seeds (run_households.simulate) matches the exact expectation within 4 Monte Carlo standard errors.
"""
import numpy as np
import pandas as pd

import willing_able as wa
import run_household_model as hm
import run_households as rh

B = np.array([-3.9, 4.0, 0.5, 0.0, 1.3])               # illustrative receipt coefficients (b_k >= 0, as calibrated)


def main():
    pop = wa.load_population()
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    T = len(inputs.dates)
    a = wa.ability(pop, inputs)

    # 1. probabilities
    d = wa.desire(pop, T, -0.23, -3.37, (0.3, 0.3))
    tables = wa.StateTables(pop, a, d)
    for kappa in (0.0, 0.5, 1.0):
        worst = max(np.abs(P.sum(1) - 1).max() for P in tables.pairs(kappa))
        assert worst < 1e-10, (kappa, worst)
    senders = np.array([(P @ np.add.outer(np.arange(pop['n_core'][h] + 1), np.arange(pop['n_kin'][h] + 1)).ravel())
                        for h, P in enumerate(tables.pairs(0.0))]).T                       # (T, H)
    expect = np.stack([np.bincount(pop['hid'], weights=(a * d)[t], minlength=pop['H']) for t in range(T)])
    assert np.abs(senders - expect).max() < 1e-9
    print('1. probabilities: ok')

    # 2. nesting in the current household model
    d1 = wa.desire(pop, T, 30.0, 30.0)
    P_new, _ = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, d1).pairs(0.0), B)
    cells = hm.Cells(pop, inputs)
    x = np.r_[B, 0.0, 0.0]          # hm TERMS: intercept, any_core, log_core, log_kin, log_kin_no_core, no_worker, any_older
    P_old = cells.receipt(x, kappa=0.0)
    gap = np.abs(P_new - P_old).max()
    assert gap < 1e-10, gap
    print(f'2. nesting in run_household_model at kappa = 0: max abs difference {gap:.2e}')

    # 3. stepping-in monotone (holds because the calibration constrains b_k >= 0), single links unaffected
    prev = None
    single = (pop['n_core'] + pop['n_kin']) == 1
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        P, Tr = wa.receipt_and_transfers(pop, tables.pairs(kappa), B)
        if prev is not None:
            assert (P >= prev[0] - 1e-12).all() and (Tr >= prev[1] - 1e-12).all()
            assert np.abs(P[:, single] - prev[0][:, single]).max() < 1e-12
        prev = (P, Tr)
    print('3. stepping-in: monotone in kappa, single links unaffected')

    # 4. exact vs simulated
    p = dict(a_c=-0.23, a_k=-3.37, eta=np.array([0.3, 0.3]), b=B)
    P_exact, _ = wa.receipt_and_transfers(pop, tables.pairs(0.8), B)
    sims = np.array([rh.simulate(pop, inputs, p, 0.8, 0.0, None, 0.0, np.zeros(T), s)[0].mean(1) for s in range(400)])
    mc_se = sims.std(0, ddof=1) / np.sqrt(len(sims))
    z = np.abs(sims.mean(0) - P_exact.mean(1)) / mc_se
    assert (z < 4).all(), z
    print(f'4. exact vs simulated mean receipt: max |z| {z.max():.2f} over {T} months')


if __name__ == '__main__':
    main()
