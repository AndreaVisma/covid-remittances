"""Small independent fixtures for CPS parsing and statistics (no full-data run)."""
from pathlib import Path
import importlib.util
import tempfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('cps', Path(__file__).with_name('cps-earnings-employment.py'))
cps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cps)
layout, labels = cps.read_layout()
width = max(a + b for a, b, _ in layout.values())


def record(**updates):
    fields = dict(YEAR=2019, MONTH=1, AGE=30, HISPAN=100, EMPSTAT=10, CLASSWKR=22,
                  MISH=4, OCC2010=6260, SEX=1, EDUC=73, EARNWEEK2=123456,
                  EARNWT=200000, WTFINL=300000, CPSIDV=201901123456789)
    fields.update(updates)
    row = bytearray(b'0' * width)
    for name, value in fields.items():
        start, size, _ = layout[name]
        row[start:start+size] = str(value).zfill(size).encode()
    return bytes(row) + b'\n'


with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'fixture.dat'
    path.write_bytes(b''.join([record(), record(EARNWEEK2=99999999), record(EARNWEEK2=0),
                               record(HISPAN=901), record(AGE=17), record(ASECFLAG=1)]))
    cps.DATA = path
    p, counts, _ = cps.load_people(layout)
    assert len(p) == 3 and counts.source_records.sum() == 6
    np.testing.assert_allclose(p.EARNWEEK2, [1234.56, 999999.99, 0.])
    np.testing.assert_allclose(p.EARNWT, 20.)
    np.testing.assert_allclose(p.WTFINL, 30.)
    assert p.CPSIDV.iloc[0] == 201901123456789
    assert p.earn_valid.tolist() == [True, False, True]
    result = cps.earnings_stats(p.loc[p.earn_valid])
    np.testing.assert_allclose(result['monthly_mean_usd'], 617.28 * 52 / 12)

np.testing.assert_array_equal(cps.weighted_quantile([100, 300], [1, 3], [.25, .5, .75]), [100, 300, 300])
g = pd.DataFrame(dict(civilian=[True]*4, WTFINL=[10., 30., 20., 40.],
                     employed=[True, True, False, False], unemployed=[False, False, True, False]))
np.testing.assert_allclose(cps.employment_stats(g)['unemployment_rate'], 1/3)
g = pd.DataFrame(dict(std_cell=[100]*3+[200]*3, EARNWEEK2=[100.]*3+[200.]*3, EARNWT=[1.]*6))
ref = pd.Series([.75, .25], index=[100, 200])
value, coverage = cps.standardized_earnings(g, ref)
np.testing.assert_allclose(value, 125 * 52/12)
assert coverage == 1.
value, coverage = cps.standardized_earnings(g.iloc[:3], ref)
assert np.isnan(value) and coverage == .75
np.testing.assert_array_equal(cps.occupation_codes(np.array([6050, 6260, 4220, 5700, 9999])), [0, 1, 2, 3, 4])
print('PASS: fixed-width scaling, IDs, sample filters, NIU vs zero, weighted statistics, composition gaps and occupation groups.')
