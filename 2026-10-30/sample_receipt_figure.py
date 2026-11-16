"""Sample figure: where households sit on the receipt curve, willing-and-able model (30 Oct 2026).

Run: python 2026-10-30/sample_receipt_figure.py      (reads outputs/calibration_simple.csv, the kappa = 0 calibration;
     no estimation; simplified forms since 13 Nov 2026)
Same illustrative households as recipient-level S1 (made-up compositions, not MMP households), all migrants
working. A household's position is the log-odds of its expected monthly receipt, averaged exactly over
which migrants want to send (Steps 2-5 of model_plan.tex). Job loss: the migrant can no longer send
(ability 0), without stepping-in (kappa = 0) and with full stepping-in (kappa = 1, capped by the reserve).
Parameters: the kappa = 0 calibration (normal times without stepping-in), as estimated.
"""
from itertools import product
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit, logit
from scipy.stats import binom

import willing_able as wa

_spec = importlib.util.spec_from_file_location('model_make_figures', wa.NCM / 'model/make_figures.py')
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

FIG = wa.HERE / 'figures'
NET_COLORS = {'Core only': mf.BLUE, 'Core + kin': mf.ORANGE, 'Kin only': mf.AQUA}
STEP_IN = mf.K4                                                  # stepping-in marker (dark blue)


def household_receipt(p, n_core, n_kin, no_worker, older, need_mean, lost=None, kappa=0.0):
    """Exact expected receipt of one household (simplified forms). lost: 'core' or 'kin' migrant who cannot send
    (ability 0). Desire expit(a_role + delta M_j + need), M_j = other core migrants; effective senders
    E = W + min(X, R), X ~ Bin(L, kappa); receipt expit(b0 + b E)."""
    need = p['eta'] @ (np.array([no_worker, older], float) - need_mean)
    d_core = expit(p['a_c'] + p['delta'] * (n_core - 1) + need)
    d_kin = expit(p['a_k'] + p['delta'] * n_core + need)
    migrants = [(True, d_core)] * n_core + [(False, d_kin)] * n_kin
    able = [1.0] * len(migrants)
    if lost is not None:
        able[[i for i, (c, _) in enumerate(migrants) if c == (lost == 'core')][0]] = 0.0
    P = 0.0
    for wants in product((0, 1), repeat=len(migrants)):          # ability is 0/1 here, so desire is the only draw
        pr = np.prod([d if w else 1 - d for (_, d), w in zip(migrants, wants)])
        W = sum(w * a for w, a in zip(wants, able))
        L = sum(w * (1 - a) for w, a in zip(wants, able))
        R = sum((1 - w) * a for w, a in zip(wants, able))
        for x in range(int(L) + 1):
            q = binom.pmf(x, L, kappa)
            if q > 0:
                P += pr * q * wa.receipt_prob(p['b'], W + min(x, R), 0)
    return P


def main():
    mf.style()
    FIG.mkdir(exist_ok=True)
    cal = wa.load_calibration()
    p = wa.params_for(cal, 0.0)
    pop = wa.load_population()
    h = pop['h']
    raw = np.column_stack([h.resident_worker.eq(0), h.resident_older.gt(0)]).astype(float)
    known = (h.resident.gt(0) & h[['resident_worker', 'resident_older']].notna().all(axis=1)).to_numpy()
    need_mean = raw[known].mean(0)

    # label, network type, (core, kin, no worker, older), label position, job loss (who, text) or None
    examples = [
        ('Two cousins in the US;\nworkers at home', 'Kin only', (0, 2, 0, 0), (-5.0, 0.24), None),
        ('Two cousins; no worker,\na grandparent at home', 'Kin only', (0, 2, 1, 1), (-1.2, 0.08), None),
        ('One son in the US; workers at home', 'Core only', (1, 0, 0, 0), (0.55, 0.30), ('core', 'he loses his job')),
        ('Three children and two cousins;\nno worker, a grandparent', 'Core + kin', (3, 2, 1, 1), (-1.6, 0.90),
         ('core', 'one child loses their job')),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.linspace(-5.2, 3.5, 300)
    ax.plot(x, expit(x), color=mf.GRAY_LIGHT, lw=2.5, zorder=1, solid_capstyle='round')
    mf.pct_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Probability of receiving in a month')
    rows = []
    for label, net, (c, k, nw, ol), pos, loss in examples:
        P0 = household_receipt(p, c, k, nw, ol, need_mean)
        s0 = logit(P0)
        text = f'{label}: {P0:.0%}'
        row = dict(household=label.replace('\n', ' '), network=net, receipt=P0)
        if loss is not None:
            who, what = loss
            P1 = household_receipt(p, c, k, nw, ol, need_mean, lost=who, kappa=0.0)
            P2 = household_receipt(p, c, k, nw, ol, need_mean, lost=who, kappa=1.0)
            s1, s2 = logit(P1), logit(P2)
            ax.annotate('', (s1, P1), (s0, P0), arrowprops=dict(arrowstyle='-|>', color=mf.RED, lw=1.6, shrinkA=6,
                                                                shrinkB=5, connectionstyle='arc3,rad=0.25'))
            ax.scatter(s1, P1, s=40, facecolor=mf.SURFACE, edgecolor=mf.RED, linewidth=1.5, zorder=5)
            if abs(P2 - P1) > 0.005:
                ax.annotate('', (s2, P2), (s1, P1), arrowprops=dict(arrowstyle='-|>', color=STEP_IN, lw=1.4,
                                                                    shrinkA=5, shrinkB=5, ls='--',
                                                                    connectionstyle='arc3,rad=-0.9'))
                ax.scatter(s2, P2, s=40, facecolor=mf.SURFACE, edgecolor=STEP_IN, linewidth=1.5, zorder=5)
                text += f'\n{what}: {P1:.0%}; someone steps in: {P2:.0%}'
            else:
                text += f'\n{what}: {P1:.0%}; nobody can step in'
            row.update(job_loss=P1, job_loss_stepping_in=P2)
        rows.append(row)
        mf.ring_dot(ax, s0, P0, NET_COLORS[net], size=70, zorder=6)
        ax.annotate(text, (s0, P0), xytext=pos, textcoords='data', fontsize=8, color=mf.INK2, va='center',
                    arrowprops=dict(arrowstyle='-', color=mf.MUTED, lw=0.7, shrinkA=2, shrinkB=6))
    ax.set_xlabel('Household score (log-odds of expected receipt)')
    ax.set_xlim(x[0], x[-1])
    handles = [Line2D([], [], marker='o', ls='', color=col, markeredgecolor=mf.SURFACE, markersize=8, label=n)
               for n, col in NET_COLORS.items()]
    handles += [Line2D([], [], color=mf.RED, lw=1.6, marker='>', markersize=5, label='One migrant loses their job'),
                Line2D([], [], color=STEP_IN, lw=1.4, ls='--', marker='>', markersize=5,
                       label='Someone steps in (κ = 1)')]
    ax.legend(handles=handles, loc='upper left', frameon=False)
    b0, b1 = p['b']
    terms = (f"receipt given effective senders E = expit({b0:.2f} + {b1:.2f}·E)\n"
             f"desire: core expit({p['a_c']:.2f} {p['delta']:+.2f}·other core migrants + need), "
             f"kin expit({p['a_k']:.2f} {p['delta']:+.2f}·core migrants + need);  "
             f"need = {p['eta'][0]:.2f}·no worker + {p['eta'][1]:.2f}·older (centred)\n"
             f"κ = 0 calibration (MMP 2019); score = log-odds of receipt averaged over who wants to send")
    ax.text(0.0, -0.17, terms, transform=ax.transAxes, fontsize=7, color=mf.MUTED, va='top')
    fig.suptitle('Where households sit on the receipt curve (willing-and-able model, sample)', x=0.01, ha='left',
                 fontsize=12.5, fontweight='semibold', color=mf.INK)
    out = FIG / 'S1_receipt_curve_examples_sample.png'
    fig.savefig(out, dpi=mf.DPI, bbox_inches='tight')
    print('saved', out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
