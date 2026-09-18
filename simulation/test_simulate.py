"""Checks on accounting, state persistence, and counterfactual comparability."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from simulate import HERE, SCENARIOS, aggregate, simulate, validate


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((HERE / "parameters.json").read_text())
        self.p["population"] = 80

    def test_budget_and_continuity_all_scenarios(self):
        for scenario in SCENARIOS:
            a, _ = simulate(self.p, scenario)
            np.testing.assert_allclose(
                a.opening_savings + a.income + a.us_transfers,
                a.consumption + a.remittances + a.closing_savings,
                atol=1e-8,
            )
            self.assertTrue((a[["closing_savings", "consumption", "remittances", "unmet_remittances"]] >= -1e-8).all().all())
            previous = a.groupby("agent_id").closing_savings.shift()
            mask = previous.notna()
            np.testing.assert_allclose(a.loc[mask, "opening_savings"], previous[mask])
            m = aggregate(a)
            np.testing.assert_allclose(m.participation * m.mean_per_sender * self.p["population"], m.remittances)
            np.testing.assert_allclose(m[["employed_share", "unemployed_share", "inactive_share"]].sum(axis=1), 1)
            self.assertEqual(len(a), 60 * self.p["population"])

    def test_reproducible_and_shared_pre_covid(self):
        baseline, _ = simulate(self.p)
        repeat, _ = simulate(self.p)
        pd.testing.assert_frame_equal(baseline, repeat)
        for scenario in SCENARIOS[1:]:
            other, _ = simulate(self.p, scenario)
            cols = baseline.columns.drop("scenario")
            before = baseline.date < "2020-03-01"
            pd.testing.assert_frame_equal(baseline.loc[before, cols], other.loc[before, cols])

    def test_no_resources_no_remittances(self):
        self.p.update(initial_state_shares=[0, 0, 1], zero_savings_share=1,
                      fiscal_eligible_share=0,
                      transition_normal=np.eye(3).tolist(), transition_crisis=np.eye(3).tolist())
        a, _ = simulate(self.p)
        self.assertEqual(a.remittances.sum(), 0)
        self.assertEqual(a.closing_savings.sum(), 0)
        self.assertTrue((a.consumption_shortfall > 0).all())
        self.assertTrue((a.state == "inactive").all())

    def test_invalid_parameters(self):
        for key, value in (("population", 0), ("remit_probability", 1), ("initial_state_shares", [1, 1, 1])):
            p = copy.deepcopy(self.p)
            p[key] = value
            with self.assertRaises(ValueError):
                validate(p)
        self.p["pandemic"].append(self.p["pandemic"][0].copy())
        with self.assertRaises(ValueError):
            validate(self.p)


if __name__ == "__main__":
    unittest.main()
