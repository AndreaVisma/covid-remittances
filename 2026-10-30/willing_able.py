"""Willing-and-able model with stepping-in: shared library (30 Oct 2026; plan: model_plan.tex).

Not a script. Used by calibrate_mmp.py (MMP 2019 calibration), run_shocks.py (shock years, compatibility
map) and run_households.py (household paths: persistence test, who gained and lost).

Per migrant j and month t (plan Steps 1-3):
    ability   a_j,t = e_o(j),t + (1 - e_o(j),t) * pi * S_t          S_t: stimulus months switched on
    desire    theta_j,t = a_type(j) + g [core] log N^c_h + eta_w no_worker_h + eta_o older_h + beta N*_t,
              d_j,t = expit(theta)   (g: core desire falls with the number of core migrants; user, 30 Oct)
    states    sender a d | lost sender (1-a) d | reserve a (1-d) | inactive (1-a)(1-d), independent draws
Per household h (Steps 4-6):
    replacements  X^c ~ Bin(L^c, kappa), X^k ~ Bin(L^k, kappa); capped by the reserve R, core roles first:
                  (kappa in (1, 2]: X = L + Bin(L, kappa - 1), each lost sender replaced for sure and a second
                  reserve migrant also sends with probability kappa - 1; exploratory, 2 Nov 2026)
                  X^c' = min(X^c, R), X^k' = min(X^k, R - X^c')
    effective     E^c = W^c + X^c', E^k = W^k + X^k'
    receipt       s_h(E^c, E^k) = expit(b0 + b_any [E^c>0] + b_c log(1+E^c) + b_k0 [N^c_h = 0] log(1+E^k))
                  (kin matter only in households without core migrants; b_k removed, user, 30 Oct)
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
B_NAMES = ['b0', 'b_any', 'b_c', 'b_k0']
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


def desire(pop, T, a_c, a_k, eta=(0.0, 0.0), beta=0.0, need_stock=None, g=0.0):
    """(T, M) d_j,t = expit(a_type + g [core] log N^c_h + eta' need_h + beta N*_t). need_x: centred household
    need features. a_c is the desire of a core migrant who is the household's only core migrant."""
    need = pop['need_x'] @ np.asarray(eta, float)                         # (H,)
    core_size = np.log(np.maximum(pop['n_core'][pop['hid']], 1))            # (M,) log N^c_h, used for core only
    base = np.where(pop['core'], a_c + g * core_size, a_k) + need[pop['hid']]   # (M,)
    stock = np.zeros(T) if need_stock is None else np.asarray(need_stock, float)
    return expit(base[None, :] + beta * stock[:, None])


# %% Step 3: count distributions per group
def count_distribution(pS, pL, pR):
    """pS, pL, pR: (..., n) state probabilities of the n migrants of one group (core or kin); leading dimensions
    (e.g. households, months) are kept. Returns (..., n+1, n+1, n+1): joint distribution of (senders W, lost
    senders L, reserve R)."""
    lead, n = pS.shape[:-1], pS.shape[-1]
    D = np.zeros(lead + (n + 1, n + 1, n + 1))
    D[..., 0, 0, 0] = 1.0
    for j in range(n):
        s, l, r = (p[..., j, None, None, None] for p in (pS, pL, pR))
        new = D * (1.0 - s - l - r)
        new[..., 1:, :, :] += D[..., :-1, :, :] * s
        new[..., :, 1:, :] += D[..., :, :-1, :] * l
        new[..., :, :, 1:] += D[..., :, :, :-1] * r
        D = new
    return D


def _triples(n):
    """All (w, l, r) with w + l + r <= n, as three index arrays."""
    w, l, r = np.meshgrid(*(np.arange(n + 1),) * 3, indexing='ij')
    keep = (w + l + r) <= n
    return w[keep], l[keep], r[keep]


def replacement_pmf(x, L, kappa):
    """P(X = x) for the number of replacements drawn for L lost senders: Bin(L, kappa) for kappa <= 1;
    L + Bin(L, kappa - 1) for 1 < kappa <= 2 (before the reserve cap)."""
    assert 0.0 <= kappa <= 2.0, kappa
    if kappa <= 1.0:
        return binom.pmf(x, L, kappa)
    return binom.pmf(x - L, L, kappa - 1.0)


def relabel(ec, ek, nc, nk):
    """Keep the total number of able senders, with at most nc in core roles and nk in kin roles (overflow moves to
    the other role). Only binds for kappa > 1, when two replacements can take the role of one lost sender."""
    tot = ec + ek
    ec2 = np.clip(ec, tot - nk, nc)
    return ec2, tot - ec2


def max_replacements(L, kappa):
    return (L.max() * (2 if kappa > 1.0 else 1)) if len(L) else 0


_MAPPING, _SHORTAGE, _PREV = {}, {}, {}


def _predecessors(n):
    """For the cells of _triples(n): indices of the cells with one fewer sender, lost sender, reserve (or K, a
    padding cell that holds zero, where the count is already 0)."""
    if n not in _PREV:
        w, l, r = _triples(n)
        K = len(w)
        index = np.full((n + 2, n + 2, n + 2), K)
        index[w, l, r] = np.arange(K)
        _PREV[n] = tuple(np.where(c > 0, index[w - (c is w), l - (c is l), r - (c is r)], K) for c in (w, l, r))
    return _PREV[n]


def count_cells(pS, pL, pR):
    """As count_distribution, but returns the probabilities of the cells _triples(n) only, (..., K), computed on
    those cells directly (the full (n+1)^3 cube is mostly impossible cells)."""
    lead, n = pS.shape[:-1], pS.shape[-1]
    pw, pl, pr = _predecessors(n)
    K = len(pw)
    D = np.zeros(lead + (K + 1,))
    D[..., 0] = 1.0                                      # cell (0, 0, 0) is first in _triples order
    for j in range(n):
        s, l, r = (x[..., j, None] for x in (pS, pL, pR))
        new = np.zeros_like(D)
        new[..., :K] = D[..., :K] * (1.0 - s - l - r) + D[..., pw] * s + D[..., pl] * l + D[..., pr] * r
        D = new
    return D[..., :K]



def size_blocks(pop):
    """Households grouped by (n_core, n_kin): list of (nc, nk, hh (G,), core members (G, nc), kin members (G, nk)).
    Cached in pop."""
    if '_size_blocks' not in pop:
        blocks = []
        for nc, nk in sorted(set(zip(pop['n_core'], pop['n_kin']))):
            hh = np.flatnonzero((pop['n_core'] == nc) & (pop['n_kin'] == nk))
            core = np.array([pop['members'][h][pop['core'][pop['members'][h]]] for h in hh]).reshape(len(hh), nc)
            kin = np.array([pop['members'][h][~pop['core'][pop['members'][h]]] for h in hh]).reshape(len(hh), nk)
            blocks.append((int(nc), int(nk), hh, core.astype(int), kin.astype(int)))
        pop['_size_blocks'] = blocks
    return pop['_size_blocks']


def _mapping(nc, nk, kappa):
    """Sparse (Kc*Kk, (nc+1)(nk+1)) map from joint (W, L, R) core x kin cells to effective able senders (E^c, E^k):
    binomial replacements capped by the reserve, core roles first. Depends only on the household size and kappa."""
    key = (nc, nk, round(float(kappa), 10))
    if key not in _MAPPING:
        cw, cl, cr = _triples(nc)
        kw, kl, kr = _triples(nk)
        R = cr[:, None] + kr[None, :]
        rows, cols, vals = [], [], []
        for xc in range(max_replacements(cl, kappa) + 1):
            bc = replacement_pmf(xc, cl, kappa)
            if not bc.any():
                continue
            xc_eff = np.minimum(xc, R)
            for xk in range(max_replacements(kl, kappa) + 1):
                w = bc[:, None] * replacement_pmf(xk, kl, kappa)[None, :]
                if not w.any():
                    continue
                xk_eff = np.minimum(xk, R - xc_eff)
                ec, ek = relabel(cw[:, None] + xc_eff, kw[None, :] + xk_eff, nc, nk)
                nz = w > 0
                rows.append(np.flatnonzero(nz.ravel()))
                cols.append((ec * (nk + 1) + ek)[nz])
                vals.append(w[nz])
        _MAPPING[key] = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                          shape=(len(cw) * len(kw), (nc + 1) * (nk + 1)))   # duplicates summed
    return _MAPPING[key]


def _shortage_cells(nc, nk, kappa):
    """(3, Kc*Kk): per joint cell, expected replacements drawn, unfilled because the reserve ran out, cap binds."""
    key = (nc, nk, round(float(kappa), 10))
    if key not in _SHORTAGE:
        cw, cl, cr = _triples(nc)
        kw, kl, kr = _triples(nk)
        R = cr[:, None] + kr[None, :]
        out = np.zeros((3,) + R.shape)
        for xc in range(max_replacements(cl, kappa) + 1):
            bc = replacement_pmf(xc, cl, kappa)
            for xk in range(max_replacements(kl, kappa) + 1):
                w = bc[:, None] * replacement_pmf(xk, kl, kappa)[None, :]
                out += w * np.stack([np.full(R.shape, xc + xk), np.maximum(xc + xk - R, 0), xc + xk > R])
        _SHORTAGE[key] = out.reshape(3, -1)
    return _SHORTAGE[key]


class StateTables:
    """Per-household distributions of (W, L, R) for core and kin, given ability a and desire d (T, M), computed
    for all households of the same size at once (2 Nov 2026; results identical to the per-household version).
    pairs(kappa) -> list over households of (T, (n_c+1)(n_k+1)) probabilities of (E^c, E^k)."""

    def __init__(self, pop, a, d):
        self.pop, self.T = pop, a.shape[0]
        pS, pL, pR = a * d, (1 - a) * d, a * (1 - d)
        self.blocks = []
        for nc, nk, hh, core, kin in size_blocks(pop):
            P = []
            for n, idx in ((nc, core), (nk, kin)):
                P.append(count_cells(*(np.moveaxis(x[:, idx], 0, 1) for x in (pS, pL, pR))))   # (G, T, K)
            self.blocks.append((nc, nk, hh, P[0], P[1]))
        self._cache = {}

    def _joint(self, Pc, Pk):
        G, T, Kc = Pc.shape
        return (Pc[:, :, :, None] * Pk[:, :, None, :]).reshape(G * T, Kc * Pk.shape[2])

    def pairs(self, kappa):
        key = round(float(kappa), 10)
        if key not in self._cache:
            out = [None] * self.pop['H']
            for nc, nk, hh, Pc, Pk in self.blocks:
                E = np.asarray(_mapping(nc, nk, kappa).T.dot(self._joint(Pc, Pk).T).T)      # (G*T, n_out)
                E = E.reshape(len(hh), self.T, -1)
                for i, h in enumerate(hh):
                    out[h] = E[i]
            self._cache[key] = out
        return self._cache[key]

    def shortage(self, kappa):
        """(3, T) totals over households: replacements drawn, replacements unfilled, households short."""
        tot = np.zeros((3, self.T))
        for nc, nk, hh, Pc, Pk in self.blocks:
            J = self._joint(Pc, Pk).reshape(len(hh), self.T, -1).sum(0)                    # (T, Kc*Kk)
            tot += _shortage_cells(nc, nk, kappa) @ J.T
        return tot


# %% Steps 5-6: receipt and transfers
def receipt_grid(b, nc, nk):
    """(nc+1, nk+1) s_h(E^c, E^k) for a household with nc core and nk kin migrants; b in B_NAMES order."""
    ec, ek = np.meshgrid(np.arange(nc + 1), np.arange(nk + 1), indexing='ij')
    b0, b_any, b_c, b_k0 = b
    z = b0 + b_any * (ec > 0) + b_c * np.log1p(ec) + b_k0 * (nc == 0) * np.log1p(ek)
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
    """(n, 4) receipt design for flattened pairs (B_NAMES order)."""
    no_core = (pop['n_core'][hh] == 0).astype(float)
    return np.column_stack([np.ones(len(hh)), (ec > 0), np.log1p(ec), no_core * np.log1p(ek)])


def receipt_and_transfers(pop, pairs, b):
    """(T, H) expected receipt P_h,t and expected transfers T_h,t (Steps 5-6), computed by household size."""
    T = pairs[0].shape[0]
    P, Tr = np.zeros((T, pop['H'])), np.zeros((T, pop['H']))
    for nc, nk, hh, _, _ in size_blocks(pop):
        s = receipt_grid(b, nc, nk).ravel()
        tot = np.maximum(np.add.outer(np.arange(nc + 1), np.arange(nk + 1)).ravel(), 1)
        W = np.stack([pairs[h] for h in hh])                                  # (G, T, n_out)
        P[:, hh] = (W @ s).T
        Tr[:, hh] = (W @ (s * tot)).T
    return P, Tr


# %% Calibration records
def load_calibration(path=None):
    """outputs/calibration.csv (calibrate_mmp.py): one row per kappa with a_c, a_k, g, eta_*, b_*."""
    return pd.read_csv(path or OUT / 'calibration.csv')


def params_for(cal, kappa):
    """Normal-times parameters used at this kappa: the row calibrated at kappa if there is one, otherwise the
    kappa = 0 calibration (the base since 30 Oct: stepping-in enters only the shock runs)."""
    rows = cal.loc[np.isclose(cal.kappa, kappa)]
    row = (rows if len(rows) else cal.loc[np.isclose(cal.kappa, 0.0)]).iloc[0]
    return dict(a_c=float(row.a_c), a_k=float(row.a_k), g=float(row.g), eta=row[ETA_NAMES].to_numpy(float),
                b=row[B_NAMES].to_numpy(float))
