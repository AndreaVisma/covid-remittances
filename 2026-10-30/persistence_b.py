"""Persistence option B: regular and occasional senders (2 Nov 2026). Library; scripts in run_persistence_b.py.

Core migrants only (kin keep independent monthly desire). Each core migrant j gets a fixed type, drawn once per
path and independently across migrants:
    regular sender     with probability lambda * d_j0   (d_j0: calibrated 2019 desire, kappa = 0 calibration, no
                       crisis, wa.desire with beta = 0); wants to send every month (sends whenever able)
    occasional sender  otherwise; wants to send in month t with probability (iid across months)
                       q_j,t = (d_j,t - lambda d_j0) / (1 - lambda d_j0),  clipped to [0, 1]
d_j,t is the current model's desire including the crisis term beta N*_t. The marginal monthly desire is then
lambda d_j0 + (1 - lambda d_j0) q_j,t = d_j,t exactly (when no clipping), so with migrants independent the exact
household computation (wa.StateTables) and the national index are those of the current model for every lambda.
lambda = 0 is the current model. Types are drawn from a seventh random stream (SeedSequence(seed).spawn(7)[6]):
the first six streams are those of run_households.simulate, so lambda = 0 reproduces it exactly, and a COVID run
and its no-COVID twin with the same seed share the same types. Types are nested across lambda (same uniform).
Nothing in willing_able.py or run_households.py is modified.
"""
import numpy as np
from scipy.stats import norm

import willing_able as wa


def base_desire(pop, p):
    """(M,) d_j0: 2019 desire, no crisis (beta = 0)."""
    return wa.desire(pop, 1, p['a_c'], p['a_k'], p['eta'], 0.0, None, p['g'])[0]


def occasional_rate(d, d0, lam):
    """(T, M) q_j,t for occasional senders and the number of (t, j) cells clipped at 0 (d_j,t < lambda d_j0)."""
    raw = (d - lam * d0[None, :]) / (1 - lam * d0[None, :])
    return np.clip(raw, 0.0, 1.0), int((raw < -1e-12).sum())


def simulate(pop, inputs, p, kappa, beta, need, pi, stim, seed, lam, return_extra=False):
    """run_households.simulate with regular / occasional core senders. Returns rec (T, H) bool, trans (T, H) int
    (and, with return_extra, a dict with the core desire draws' monthly mean, the expected d, types, clip count)."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
    phi = wa.abm.employment_phi()
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['g'])
    d0 = base_desire(pop, p)
    regular = core & (streams[6].random(M) < lam * d0)
    q, n_clip = occasional_rate(d, d0, lam)
    prob = np.where(core[None, :], q, d)                     # occasional core: q; kin: d (regular overridden)
    nc, nk = pop['n_core'], pop['n_kin']
    rec, trans = np.zeros((T, H), bool), np.zeros((T, H), int)
    z = streams[0].standard_normal(M)
    hsum = lambda v: np.bincount(hid, weights=v, minlength=H).astype(int)
    want_core_mean = np.zeros(T)
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        employed = norm.cdf(z) < inputs.rates[t, pop['job_index']]
        checked = streams[2].random(M) < pi * stim[t]
        able = (employed | checked) & pop['eligible']
        wants = (streams[3].random(M) < prob[t]) | regular
        want_core_mean[t] = wants[core].mean()
        lost = wants & ~able
        u = streams[4].random(M)
        cand = lost * ((u < kappa).astype(int) if kappa <= 1 else 1 + (u < kappa - 1))
        Wc, Wk = hsum(wants & able & core), hsum(wants & able & ~core)
        Xc, Xk, R = hsum(cand * core), hsum(cand * ~core), hsum(able & ~wants)
        xc = np.minimum(Xc, R)
        xk = np.minimum(Xk, R - xc)
        Ec, Ek = wa.relabel(Wc + xc, Wk + xk, nc, nk)
        b0, b_any, b_c, b_k0 = p['b']
        s = wa.expit(b0 + b_any * (Ec > 0) + b_c * np.log1p(Ec) + b_k0 * (nc == 0) * np.log1p(Ek))
        rec[t] = streams[5].random(H) < s
        trans[t] = rec[t] * np.maximum(Ec + Ek, 1)
    if return_extra:
        return rec, trans, dict(want_core_mean=want_core_mean, d_core_mean=d[:, core].mean(1), regular=regular,
                                n_clip=n_clip)
    return rec, trans


def ever_sends(pop, inputs, p, lam, lengths, seeds, seed0=11000):
    """mig174 analogue. Core migrants (eligible), 2019 conditions (inputs: 2019 calendar months repeated), no
    crisis, no stimulus. For each L in lengths: probability that the migrant wants to send AND is able in at least
    one of L consecutive months (trip starts in a random calendar month; employment stationary at the start).
    Returns {L: mean over core migrants and seeds}."""
    core = pop['core'] & pop['eligible']
    idx = np.flatnonzero(core)
    d0 = base_desire(pop, p)[idx]
    q = d0 * (1 - lam) / (1 - lam * d0)                      # occasional rate without crisis
    phi = wa.abm.employment_phi()[idx] if np.ndim(wa.abm.employment_phi()) else wa.abm.employment_phi()
    rates = inputs.rates[:, pop['job_index'][idx]]           # (12, n)
    Lmax = max(lengths)
    out = {L: 0.0 for L in lengths}
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        n = len(idx)
        regular = rng.random(n) < lam * d0
        start = rng.integers(0, 12, n)
        z = rng.standard_normal(n)
        done = np.zeros(n, bool)
        for t in range(Lmax):
            if t > 0:
                z = phi * z + np.sqrt(1 - phi ** 2) * rng.standard_normal(n)
            able = norm.cdf(z) < rates[(start + t) % 12, np.arange(n)]
            wants = regular | (rng.random(n) < q)
            done |= wants & able
            if t + 1 in out:
                out[t + 1] += done.mean() / seeds
    return out
