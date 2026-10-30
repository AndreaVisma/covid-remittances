"""Willing-and-able model with stepping-in: shared library (30 Oct 2026; plan: model_plan.tex).

Not a script. Used by calibrate_mmp.py (MMP 2019 calibration), run_shocks.py (shock years, compatibility
map) and run_households.py (household paths: persistence test, who gained and lost).

Per migrant j and month t (plan Steps 1-3):
    ability   a_j,t = e_o(j),t + (1 - e_o(j),t) * pi * S_t          S_t: stimulus months switched on
    desire    theta_j,t = a_type(j) + eta_w no_worker_h + eta_o older_h + beta N*_t,   d_j,t = expit(theta)
    states    sender a d | lost sender (1-a) d | reserve a (1-d) | inactive (1-a)(1-d), independent draws
Per household h (Steps 4-6):
    replacements  X^c ~ Bin(L^c, kappa), X^k ~ Bin(L^k, kappa); capped by the reserve R, core roles first:
                  X^c' = min(X^c, R), X^k' = min(X^k, R - X^c')
    effective     E^c = W^c + X^c', E^k = W^k + X^k'
    receipt       s_h(E^c, E^k) = expit(b0 + b_any [E^c>0] + b_c log(1+E^c) + b_k log(1+E^k)
                                        + b_k0 [N^c_h = 0] log(1+E^k))
    transfers     T_h,t = E[ s_h(E) * max(E^c + E^k, 1) ]
Exact for rho = 0: per household, a recursion over its core migrants gives the distribution of (W^c, L^c, R^c)
and over its kin migrants that of (W^k, L^k, R^k); the two groups are independent; the binomial replacement
and the cap are then applied exactly. The recursion does not depend on kappa, so StateTables caches it and
pairs(kappa) is cheap to call for several kappas.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
NCM = HERE.parent / 'network-compatibility-model'
sys.path[:0] = [str(NCM / 'model'), str(NCM / 'recipient-level')]
import abm                                  # noqa: E402
import run_e0_baseline as e0                # noqa: E402
import compare_household_logit as hl        # noqa: E402

OUT = HERE / 'outputs'
B_NAMES = ['b0', 'b_any', 'b_c', 'b_k', 'b_k0']
ETA_NAMES = ['eta_w', 'eta_o']
STIMULUS_ROUNDS = ('2020-04', '2021-01', '2021-03')      # the three US stimulus rounds (benefit schedule)


# %% Population
def load_population():
    """MMP household-migrant networks (abm.load_population: 'Not working' = unknown occupation, all eligible)."""
    pop = hl.load_population()
    pop['n_core'] = np.bincount(pop['hid'], weights=pop['core'], minlength=pop['H']).astype(int)
    pop['n_kin'] = np.bincount(pop['hid'], weights=~pop['core'], minlength=pop['H']).astype(int)
    pop['members'] = [np.flatnonzero(pop['hid'] == h) for h in range(pop['H'])]
    return pop


def target(pop):
    """2019 calibration target: households with complete records and known receipt (e0.target_mask)."""
    h = pop['h']
    return e0.target_mask(h), h.remus.eq(1).to_numpy(float)


# %% Steps 1-2: ability and desire
def stimulus_indicator(dates, rounds=STIMULUS_ROUNDS):
    """(T,) S_t = 1 in the stimulus months switched on (subset of STIMULUS_ROUNDS)."""
    assert set(rounds) <= set(STIMULUS_ROUNDS), rounds
    return np.isin(pd.DatetimeIndex(dates).strftime('%Y-%m'), list(rounds)).astype(float)


def ability(pop, inputs, pi=0.0, stim=None):
    """(T, M) a_j,t = e + (1 - e) pi S_t (eligible migrants; e = CPS rate of the migrant's occupation)."""
    e = abm.able_probability(pop, inputs, abm.Params(0.0, 0.0))
    if stim is None or pi == 0:
        return e
    return e + (1 - e) * pi * np.asarray(stim, float)[:, None] * pop['eligible'][None, :]


def desire(pop, T, a_c, a_k, eta=(0.0, 0.0), beta=0.0, need_stock=None):
    """(T, M) d_j,t = expit(a_type + eta' need_h + beta N*_t). need_x: centred household need features."""
    need = pop['need_x'] @ np.asarray(eta, float)                         # (H,)
    base = np.where(pop['core'], a_c, a_k) + need[pop['hid']]               # (M,)
    stock = np.zeros(T) if need_stock is None else np.asarray(need_stock, float)
    return expit(base[None, :] + beta * stock[:, None])


# %% Step 3: count distributions per group
def count_distribution(pS, pL, pR):
    """pS, pL, pR: (T, n) state probabilities of the n migrants of one group (core or kin).
    Returns (T, n+1, n+1, n+1): joint distribution of (senders W, lost senders L, reserve R)."""
    T, n = pS.shape
    D = np.zeros((T, n + 1, n + 1, n + 1))
    D[:, 0, 0, 0] = 1.0
    for j in range(n):
        s, l, r = (p[:, j, None, None, None] for p in (pS, pL, pR))
        new = D * (1.0 - s - l - r)
        new[:, 1:] += D[:, :-1] * s
        new[:, :, 1:] += D[:, :, :-1] * l
        new[:, :, :, 1:] += D[:, :, :, :-1] * r
        D = new
    return D


def _triples(n):
    """All (w, l, r) with w + l + r <= n, as three index arrays."""
    w, l, r = np.meshgrid(*(np.arange(n + 1),) * 3, indexing='ij')
    keep = (w + l + r) <= n
    return w[keep], l[keep], r[keep]


class StateTables:
    """Per-household distributions of (W, L, R) for core and kin, given ability a and desire d (T, M).
    pairs(kappa) -> list over households of (T, (n_c+1)(n_k+1)) probabilities of (E^c, E^k)."""

    def __init__(self, pop, a, d):
        self.pop, self.T = pop, a.shape[0]
        pS, pL, pR = a * d, (1 - a) * d, a * (1 - d)
        self.groups = []
        for h, mem in enumerate(pop['members']):
            core = mem[pop['core'][mem]]
            kin = mem[~pop['core'][mem]]
            g = []
            for idx in (core, kin):
                n = len(idx)
                D = count_distribution(pS[:, idx], pL[:, idx], pR[:, idx])
                w, l, r = _triples(n)
                g.append((n, w, l, r, D[:, w, l, r]))
            self.groups.append(g)
        self._cache = {}

    def household_pairs(self, h, kappa):
        (nc, cw, cl, cr, Pc), (nk, kw, kl, kr, Pk) = self.groups[h]
        R = cr[:, None] + kr[None, :]                                        # (Kc, Kk)
        n_out = (nc + 1) * (nk + 1)
        rows, cols, vals = [], [], []
        for xc in range(cl.max() + 1 if len(cl) else 1):
            bc = binom.pmf(xc, cl, kappa)                                     # (Kc,)
            if not bc.any():
                continue
            xc_eff = np.minimum(xc, R)
            for xk in range(kl.max() + 1 if len(kl) else 1):
                bk = binom.pmf(xk, kl, kappa)                                 # (Kk,)
                w = bc[:, None] * bk[None, :]
                if not w.any():
                    continue
                xk_eff = np.minimum(xk, R - xc_eff)
                out = (cw[:, None] + xc_eff) * (nk + 1) + (kw[None, :] + xk_eff)
                nz = w > 0
                rows.append(np.flatnonzero(nz.ravel()))
                cols.append(out[nz])
                vals.append(w[nz])
        Kc, Kk = len(cw), len(kw)
        M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(Kc * Kk, n_out))                         # duplicates are summed
        joint = (Pc[:, :, None] * Pk[:, None, :]).reshape(self.T, Kc * Kk)    # (T, Kc*Kk)
        return np.asarray(joint @ M)                                          # (T, n_out)

    def pairs(self, kappa):
        key = round(float(kappa), 10)
        if key not in self._cache:
            self._cache[key] = [self.household_pairs(h, kappa) for h in range(self.pop['H'])]
        return self._cache[key]


# %% Steps 5-6: receipt and transfers
def receipt_grid(b, nc, nk):
    """(nc+1, nk+1) s_h(E^c, E^k) for a household with nc core and nk kin migrants; b in B_NAMES order."""
    ec, ek = np.meshgrid(np.arange(nc + 1), np.arange(nk + 1), indexing='ij')
    b0, b_any, b_c, b_k, b_k0 = b
    z = b0 + b_any * (ec > 0) + b_c * np.log1p(ec) + b_k * np.log1p(ek) + b_k0 * (nc == 0) * np.log1p(ek)
    return expit(z)


def flatten(pop, pairs, months=None):
    """Concatenate per-household pair probabilities: returns hh (n,), ec (n,), ek (n,), w (T or 1, n).
    months: None keeps all months; 'mean' averages over months (2019 calibration)."""
    hh, ec, ek, w = [], [], [], []
    for h, P in enumerate(pairs):
        nc, nk = pop['n_core'][h], pop['n_kin'][h]
        e_c, e_k = np.divmod(np.arange((nc + 1) * (nk + 1)), nk + 1)
        hh.append(np.full(len(e_c), h)), ec.append(e_c), ek.append(e_k)
        w.append(P.mean(0, keepdims=True) if months == 'mean' else P)
    return np.concatenate(hh), np.concatenate(ec), np.concatenate(ek), np.concatenate(w, axis=1)


def design(pop, hh, ec, ek):
    """(n, 5) receipt design for flattened pairs (B_NAMES order)."""
    no_core = (pop['n_core'][hh] == 0).astype(float)
    return np.column_stack([np.ones(len(hh)), (ec > 0), np.log1p(ec), np.log1p(ek), no_core * np.log1p(ek)])


def receipt_and_transfers(pop, pairs, b):
    """(T, H) expected receipt P_h,t and expected transfers T_h,t (Steps 5-6)."""
    T = pairs[0].shape[0]
    P, Tr = np.zeros((T, pop['H'])), np.zeros((T, pop['H']))
    for h, W in enumerate(pairs):
        nc, nk = pop['n_core'][h], pop['n_kin'][h]
        s = receipt_grid(b, nc, nk).ravel()
        tot = np.add.outer(np.arange(nc + 1), np.arange(nk + 1)).ravel()
        P[:, h] = W @ s
        Tr[:, h] = W @ (s * np.maximum(tot, 1))
    return P, Tr


# %% Calibration records
def load_calibration(path=None):
    """outputs/calibration.csv (calibrate_mmp.py): one row per kappa with a_c, a_k, eta_*, b_*."""
    return pd.read_csv(path or OUT / 'calibration.csv')


def params_for(cal, kappa):
    row = cal.loc[np.isclose(cal.kappa, kappa)].iloc[0]
    return dict(a_c=float(row.a_c), a_k=float(row.a_k), eta=row[ETA_NAMES].to_numpy(float),
                b=row[B_NAMES].to_numpy(float))
