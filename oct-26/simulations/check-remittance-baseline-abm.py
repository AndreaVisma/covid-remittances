"""Focused mechanism and saved-output checks; run after remittance-baseline-abm.py."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('abm', Path(__file__).with_name('remittance-baseline-abm.py'))
abm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abm)


def fixture(households=1, budget=100.):
    return dict(n_h=households, n_m=2 * households,
        hid=np.repeat(np.arange(households), 2), core=np.ones(2 * households, dtype=bool),
        sizes=np.full(households, 2), offsets=np.arange(households) * 2,
        capacity=np.full(2 * households, budget), fraction=np.ones(2 * households),
        need=np.full(households, 100.), fixed_theta=np.zeros(2 * households),
        no_worker=np.zeros(households), calibration_id=np.zeros(households, dtype=int),
        n_groups=1, network_type=np.full(households, 'Core only'))


# A contribution meeting the target crowds out the second migrant completely.
r = abm.simulate([100., 100., 0.], fixture(), 2, 1, 0, collect=True)
assert all(x[1][0] == 1 and x[0][0] == 100. for x in r['household'])
# Budget-constrained migrants can jointly contribute; the residual is conserved.
r = abm.simulate([100., 100., 0.], fixture(budget=40.), 1, 1, 0, collect=True)
assert r['household'][0][0][0] == 80. and r['household'][0][2][0] == 20.
# A different network member steps in when the incumbent loses all capacity.
r = abm.simulate([100., 100., 0.], fixture(), 2, 1, 0, collect=True,
                 capacity_path=np.array([[1., 0.], [0., 1.]]))
np.testing.assert_array_equal(r['transfers'], [[100., 0.], [0., 100.]])
assert r['monthly'].replacement_households.iloc[1] == 1
r = abm.simulate([100., 100., 0.], fixture(budget=0.), 1, 1, 0, collect=True)
assert r['monthly'].active_senders.iloc[0] == 0 and r['moments'][0] == 0.
# With two conditional p=.4 decisions and complete crowding out, receipt and
# expected sender count both equal .64, not the independent sender count .8.
logit = np.log(.4 / .6)
r = abm.simulate([logit, logit, 0.], fixture(20000), 1, 72, 0, collect=True)
np.testing.assert_allclose(r['moments'], [.64], atol=1e-10)
assert abs(r['monthly'].receipt_rate.iloc[0] - .64) < .015
assert r['monthly'].active_senders.iloc[0] == np.count_nonzero(r['household'][0][0])
repeat = abm.simulate([logit, logit, 0.], fixture(20000), 1, 72, 0, collect=True)
np.testing.assert_array_equal(r['transfers'], repeat['transfers'])

# Read exported files back and verify agent -> household -> monthly accounting.
h = pd.read_parquet(abm.OUT / 'household_months.parquet')
m = pd.read_parquet(abm.OUT / 'migrant_months.parquet')
totals = m.groupby(['month', 'synthetic_household_id']).sent_usd.sum()
actual = h.set_index(['month', 'synthetic_household_id']).received_usd
np.testing.assert_allclose(totals.sort_index(), actual.sort_index())
summary = pd.read_csv(abm.OUT / 'monthly_summary.csv')
first = summary.loc[summary.seed.eq(abm.SEEDS[0])].set_index('month')
np.testing.assert_allclose(h.groupby('month').received_usd.sum(), first.total_usd)
np.testing.assert_allclose(m.groupby('month').sent.sum(), first.active_senders)
assert not h.duplicated(['month', 'synthetic_household_id']).any()
assert not m.duplicated(['month', 'synthetic_migrant_id']).any()
assert len(pd.read_json(abm.OUT / 'monthly_summary.json')) == len(summary)
print('PASS: crowding out, partial coverage, replacement, zero capacity, conditional probability, reproducibility and exported accounting.')
