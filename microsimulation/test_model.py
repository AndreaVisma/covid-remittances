"""Accounting, behaviour, input validation, and calibration leakage tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from data import generate_inputs, load_inputs
from model import allocate_support, simulate
from run import fit_pre_covid


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = json.loads((Path(__file__).parent / 'config.json').read_text())
        self.config.update(n_agents=30, start='2019-01-01', end='2020-12-01', seeds=[11])
        generate_inputs(self.config, self.tmp.name)
        self.inputs = load_inputs(self.tmp.name, self.config)

    def test_reproducible_and_budget_accounting(self):
        monthly, trace = simulate(self.inputs, self.config, 42, detail=True)
        pd.testing.assert_frame_equal(monthly, simulate(self.inputs, self.config, 42)[0])
        self.assertTrue((trace.sent_usd <= trace.capacity_usd + 1e-9).all())
        self.assertTrue(np.allclose(monthly.total_usd + monthly.unmet_usd, monthly.desired_usd))
        actual = (trace.sent_usd * trace.weight).groupby(trace.date).sum()
        self.assertTrue(np.allclose(actual, monthly.total_usd))

    def test_recipient_shortfall_cannot_be_paid_twice(self):
        support, unmet = allocate_support(np.array([0, 0, 0, 1]), np.array([100, 0, 0, 50]), np.array([0, 100, 300, 0]))
        np.testing.assert_allclose(support, [0, 25, 75, 0])
        np.testing.assert_allclose(unmet, [0, 50])

    def test_income_does_not_change_willingness(self):
        _, before = simulate(self.inputs, self.config, 11, detail=True)
        a, e, h = [x.copy() for x in self.inputs]
        e['wage_multiplier'] = 0.0
        monthly, after = simulate((a, e, h), self.config, 11, detail=True)
        np.testing.assert_array_equal(before.willing, after.willing)
        self.assertEqual(monthly.total_usd.sum(), 0)
        self.assertGreater(monthly.willing_count.sum(), 0)

    def test_disabled_mechanisms_and_baseline_formula(self):
        c = copy.deepcopy(self.config)
        c['mechanisms'] = dict(need=False, persistence=False, network=False)
        c['target_mode'] = 'current'
        monthly, trace = simulate(self.inputs, c, 11, detail=True)
        np.testing.assert_allclose(trace.sent_usd, np.minimum(trace.capacity_usd, trace.willing * c['remittance_share'] * trace.income_usd))
        self.assertEqual(monthly.network_usd.sum(), 0)
        c.update(need_effect=99, persistence_effect=99, support_probability=1)
        pd.testing.assert_frame_equal(monthly, simulate(self.inputs, c, 11)[0])

    def test_zero_support_matches_target_only(self):
        c = copy.deepcopy(self.config)
        c['support_probability'] = 0
        first = simulate(self.inputs, c, 11)[0]
        c['mechanisms']['network'] = False
        pd.testing.assert_frame_equal(first, simulate(self.inputs, c, 11)[0])

    def test_missing_input_month_rejected(self):
        path = Path(self.tmp.name) / 'needs.csv'
        pd.read_csv(path).iloc[1:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            load_inputs(self.tmp.name, self.config)

    def test_generation_never_overwrites_inputs(self):
        with self.assertRaisesRegex(ValueError, 'not empty'):
            generate_inputs(self.config, self.tmp.name)

    def test_calibration_ignores_holdout(self):
        target = simulate(self.inputs, self.config, 11)[0][['date', 'total_usd']].rename(columns={'total_usd': 'observed_usd'})
        fit, scores = fit_pre_covid(self.inputs, self.config, target)
        target.loc[target.date >= self.config['holdout_start'], 'observed_usd'] *= 1000
        fit2, scores2 = fit_pre_covid(self.inputs, self.config, target)
        self.assertEqual(fit, fit2)
        pd.testing.assert_frame_equal(scores, scores2)


if __name__ == '__main__':
    unittest.main()
