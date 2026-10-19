"""Mechanism fixtures and exported-panel checks for the minimal CPS ABM."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('abm',Path(__file__).with_name('remittance-minimal-cps-abm.py'))
abm=importlib.util.module_from_spec(spec); spec.loader.exec_module(abm)


def fixture(H=1, capacity=500., seed=21):
    M=2*H
    d=dict(H=H,M=M,hid=np.repeat(np.arange(H),2),core=np.ones(M,bool),adult=np.ones(M,bool),
           eligible=np.ones(M,bool),x=np.zeros((H,3)),sizes=np.full(H,2),offsets=np.arange(H)*2,
           masks=np.ones((1,H))/H)
    rng=np.random.default_rng(seed)
    bank=dict(choice=rng.random((2,M)),tie=rng.random((2,M)),capacity=np.full((2,M),capacity),
              employed=np.ones((2,M),bool))
    return d,bank


def check_mechanism():
    # Living allowance excludes low-paid employed migrants from capacity.
    floor=abm.MONTHLY_LIVING_ALLOWANCE_USD
    np.testing.assert_allclose(abm.remittable_budget([0.,floor,floor+1000.]),
                               [0.,0.,1000*abm.REMITTABLE_SURPLUS_SHARE])
    # Two willing migrants each cover the full target without coordination.
    d,b=fixture(); p=np.array([100.,100.,0.,0.,0.,0.,np.log(100.)])
    independent=abm.simulate(p,d,b,'uncoordinated',0,collect=True)
    coordinated=abm.simulate(p,d,b,'coordinated',0,collect=True)
    np.testing.assert_allclose(np.stack(independent['household'])[:,:,0],200.)
    np.testing.assert_allclose(np.stack(coordinated['household'])[:,:,0],100.)
    assert coordinated['summary'].crowded_out.eq(1).all()
    # Partial budgets permit joint contributions up to the remaining target.
    d,b=fixture(capacity=40.)
    r=abm.simulate(p,d,b,'coordinated',0,collect=True)
    np.testing.assert_allclose(np.stack(r['household'])[:,:,0],80.)
    # Incumbent capacity loss can be replaced by the other link.
    d,b=fixture(); b['capacity']=np.array([[100.,0.],[0.,100.]])
    b['employed']=b['capacity']>0
    r=abm.simulate(p,d,b,'coordinated',0,collect=True)
    np.testing.assert_allclose(r['transfers'],[[100.,0.],[0.,100.]])
    assert r['summary'].replacement_households.iloc[1]==1
    # Unemployed willingness survives; no positive transfer is possible.
    d,b=fixture(capacity=0.)
    r=abm.simulate(p,d,b,'uncoordinated',0,collect=True)
    assert r['summary'].senders.eq(0).all() and r['summary'].willing.eq(2).all()
    # With no persistence, two p=.4 decisions imply q=.64 in either mechanism.
    d,b=fixture(H=15000); p[:2]=np.log(.4/.6)
    a=abm.simulate(p,d,b,'uncoordinated',0,collect=True)
    c=abm.simulate(p,d,b,'coordinated',0,collect=True)
    np.testing.assert_allclose(a['moments'],.64,atol=1e-10)
    np.testing.assert_allclose(c['moments'],a['moments'])
    assert abs(c['summary'].receipt_rate.mean()-.64)<.01
    assert a['summary'].senders.mean()>c['summary'].senders.mean()
    # Changing positive budgets does not alter desire or q with rho=0.
    b['capacity'][:]=20.
    small=abm.simulate(p,d,b,'uncoordinated',0,collect=True)
    np.testing.assert_array_equal(a['summary'].willing,small['summary'].willing)
    np.testing.assert_allclose(a['moments'],small['moments'])
    repeat=abm.simulate(p,d,b,'uncoordinated',0,collect=True)
    np.testing.assert_array_equal(small['transfers'],repeat['transfers'])
    print('PASS: coordination, overcoverage, budgets, replacement, willingness, conditional probabilities and reproducibility.')


def check_exports():
    summary=pd.read_csv(abm.OUT/'monthly_summary.csv')
    for mode in abm.MODES:
        h=pd.read_parquet(abm.OUT/f'baseline_{mode}_household_months.parquet')
        m=pd.read_parquet(abm.OUT/f'baseline_{mode}_migrant_months.parquet')
        assert not m.duplicated(['month','synthetic_migrant_id']).any()
        totals=m.groupby(['month','synthetic_household_id']).transfer_usd.sum()
        actual=h.set_index(['month','synthetic_household_id']).received_usd
        np.testing.assert_allclose(totals.sort_index(),actual.sort_index(),atol=1e-6)
        s=summary.loc[summary['mode'].eq(mode)&summary.seed.eq(abm.SEEDS[0])].set_index('month')
        np.testing.assert_allclose(h.groupby('month').received_usd.sum(),s.total_usd)
        np.testing.assert_allclose(m.groupby('month').sent.sum(),s.senders)
        assert not (m.sent & ~m.employed).any()
        if mode=='coordinated': assert (h.received_usd<=h.perceived_need_usd+1e-5).all()
    print('PASS: first-seed agent -> household -> monthly exported accounting and employment constraints.')


if __name__=='__main__':
    check_mechanism()
    if (abm.OUT/'monthly_summary.csv').exists(): check_exports()
