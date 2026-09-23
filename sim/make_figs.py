# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""Figures for the report from results/*.npz and results.json"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results', 'data'); FIG = os.path.join(HERE, '..', 'results', 'fig')
MEDIA = os.path.join(HERE, '..', 'media'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'lines.linewidth': 0.9, 'figure.dpi': 150})
R = json.load(open(os.path.join(RES, 'results.json')))
M = np.load(os.path.join(RES, 'main.npz'))
t = M['t']; xt = M['x_true']; xh = M['x_hat']; wh = M['w_hat']; s = M['s']; fc = M['f_cmd']; ft = M['f_true']; tm = M['tmpc']; te = M['tmhe']; dtrue = M['dist_true']
def vbody(x):
    c, sn = np.cos(x[:, 5]), np.sin(x[:, 5]); return np.stack([c * x[:, 6] + sn * x[:, 7], -sn * x[:, 6] + c * x[:, 7]], 1)
vb_t, vb_h = vbody(xt), vbody(xh)
vcmd = np.minimum(0.4, np.maximum(0, 0.4 * (t - 1.0)))
shade = lambda ax: (ax.axvspan(3.5, 4.3, color='red', alpha=0.08, lw=0), ax.axvspan(5.0, t[-1], color='blue', alpha=0.05, lw=0))

# ---------------- Fig 1: closed-loop tracking
fig, ax = plt.subplots(2, 2, figsize=(7.2, 3.6), sharex=True); ax = ax.ravel()
ax[0].plot(t, vb_t[:, 0], 'k', label='true'); ax[0].plot(t, vb_h[:, 0], 'C0', lw=0.7, label='NMHE'); ax[0].plot(t, vcmd, 'C3--', label='command'); ax[0].set_ylabel('$v_{fwd}$ [m/s]'); ax[0].legend(ncol=3, loc='lower right')
ax[1].plot(t, vb_t[:, 1], 'k'); ax[1].plot(t, vb_h[:, 1], 'C0', lw=0.7); ax[1].set_ylabel('$v_{lat}$ [m/s]')
ax[2].plot(t, xt[:, 3], 'k', label='roll'); ax[2].plot(t, xt[:, 4], 'C1', label='pitch'); ax[2].plot(t, xh[:, 3], 'C0', lw=0.6); ax[2].plot(t, xh[:, 4], 'C2', lw=0.6); ax[2].set_ylabel('attitude [rad]'); ax[2].legend(ncol=2)
ax[3].plot(t, xt[:, 11], 'k', label='true'); ax[3].plot(t, xh[:, 11], 'C0', lw=0.7, label='NMHE'); ax[3].plot(t, np.where(t >= 5.0, 0.3, 0.0), 'C3--', label='command'); ax[3].set_ylabel('$\\omega_z$ [rad/s]'); ax[3].set_xlabel('time [s]'); ax[2].set_xlabel('time [s]'); ax[3].legend(ncol=3)
for a in ax: shade(a); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'tracking.pdf')); plt.close(fig)

# ---------------- Fig 2: estimation errors
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), sharex=True)
ax[0].plot(t, (xh - xt)[:, 2] * 100, 'C0', label='$z$ [cm]'); ax[0].plot(t, (xh - xt)[:, 0] * 100, 'C1', lw=0.6, label='$x$ [cm]'); ax[0].set_ylabel('position err'); ax[0].legend(ncol=2)
ax[1].plot(t, (xh - xt)[:, 6:9]); ax[1].set_ylabel('velocity err [m/s]'); ax[1].legend(['$v_x$', '$v_y$', '$v_z$'], ncol=3)
ax[2].plot(t, (xh - xt)[:, 3:6] * 1e3); ax[2].set_ylabel('attitude err [mrad]'); ax[2].legend(['roll', 'pitch', 'yaw'], ncol=3)
for a in ax: a.set_xlabel('time [s]')
for a in ax: shade(a); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'est_err.pdf')); plt.close(fig)

# ---------------- Fig 3: disturbance estimate
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), sharex=True)
ax[0].plot(t, dtrue[:, 1], 'k--', label='applied push $F_y$'); ax[0].plot(t, wh[:, 1], 'C0', label='$\\hat d_{f,y}$'); ax[0].plot(t, wh[:, 0], 'C1', lw=0.6, label='$\\hat d_{f,x}$'); ax[0].plot(t, wh[:, 2], 'C2', lw=0.6, label='$\\hat d_{f,z}$'); ax[0].set_ylabel('force [N]'); ax[0].legend(ncol=2, fontsize=6)
ax[1].plot(t, wh[:, 3:6]); ax[1].set_ylabel('torque [Nm]'); ax[1].legend(['$\\hat d_{\\tau,x}$', '$\\hat d_{\\tau,y}$', '$\\hat d_{\\tau,z}$'], ncol=3, fontsize=6); ax[1].set_xlabel('time [s]'); ax[0].set_xlabel('time [s]')
for a in ax: shade(a); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'disturbance.pdf')); plt.close(fig)

# ---------------- Fig 4: GRF commands vs realised, one gait cycle
k0, k1 = 220, 292
fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
legs = ['LF', 'RF', 'LH', 'RH']
for li in range(4):
    ax[0].plot(t[k0:k1], fc[k0:k1, li, 2], f'C{li}', label=legs[li]); ax[0].plot(t[k0:k1], ft[k0:k1, li, 2], f'C{li}', ls=':', lw=0.7)
    ax[1].plot(t[k0:k1], fc[k0:k1, li, 0], f'C{li}'); ax[1].plot(t[k0:k1], ft[k0:k1, li, 0], f'C{li}', ls=':', lw=0.7)
ax[0].set_ylabel('$f_z$ [N]'); ax[0].legend(ncol=4); ax[0].axhline(400, color='gray', ls='--', lw=0.6); ax[0].text(t[k0] + 0.01, 385, 'bound $f_{max}$', fontsize=6, color='gray')
ax[1].set_ylabel('$f_x$ [N]'); ax[1].set_xlabel('time [s]')
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'grf.pdf')); plt.close(fig)

# ---------------- Fig 5: timing
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9))
for arr, lab, c in [(tm[:, 0], 'NMPC solver prep', 'C0'), (tm[:, 1], 'NMPC solver feedback', 'C1'), (te[:, 0], 'MHE solver prep', 'C2'), (te[:, 2], 'MHE solver feedback', 'C3')]:
    x = np.sort(arr); ax[0].plot(x, np.linspace(0, 100, len(x)), c, label=lab)
ax[0].set_xscale('log'); ax[0].set_xlabel('wall time [$\\mu$s]'); ax[0].set_ylabel('CDF [%]'); ax[0].legend(fontsize=5.5); ax[0].grid(alpha=0.3)
tt = np.arange(len(tm)) * 0.01
ax[1].plot(tt, tm[:, 0] + tm[:, 1], 'C0', lw=0.6, label='NMPC solver'); ax[1].plot(t, te[:, 0] + te[:, 2], 'C2', lw=0.6, label='MHE solver'); ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('cycle [$\\mu$s]'); ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3); shade(ax[1])
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'timing.pdf')); plt.close(fig)

# ---------------- Fig 6: convergence at fixed parameter
fig, ax = plt.subplots(figsize=(3.5, 1.9))
for lab, res in R['convergence'].items(): ax.semilogy(np.arange(1, len(res) + 1), res, 'o-', ms=3, label=lab)
ax.set_xlabel('NMPC solver cycles at fixed $\\hat x$'); ax.set_ylabel('KKT residual $\\|F\\|_\\infty$'); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'convergence.pdf')); plt.close(fig)

# ---------------- Fig 7: push-recovery ablation (lateral position / velocity)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True)
for name, lab, c in [('nocomp', 'no compensation', 'C3'), ('forcecomp', 'force feed-forward', 'C0'), ('fullcomp', 'full wrench feed-forward', 'C1')]:
    D = np.load(os.path.join(RES, name + '.npz')); ax[0].plot(D['t'], D['x_true'][:, 1], c, label=lab); ax[1].plot(D['t'], D['x_true'][:, 3], c)
ax[0].set_ylabel('$p_y$ [m]'); ax[0].legend(fontsize=6); ax[1].set_ylabel('roll [rad]'); ax[1].set_xlabel('time [s]')
for a in ax: shade(a); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'ablation.pdf')); plt.close(fig)

# ---------------- Fig 8: NMPC horizon snapshot (contact schedule + predicted z, forces)
H = np.load(os.path.join(RES, 'horizon_snapshot.npz')); X, U, par = H['X'], H['U'], H['par']
fig, ax = plt.subplots(3, 1, figsize=(3.5, 3.6), sharex=True)
kk = np.arange(len(X)) * 0.01
for li in range(4): ax[0].fill_between(kk, li, li + par[:, 12 + li] * 0.8, color=f'C{li}', alpha=0.6, step='post')
ax[0].set_yticks([0.4, 1.4, 2.4, 3.4]); ax[0].set_yticklabels(legs); ax[0].set_ylabel('stance')
ax[1].plot(kk, X[:, 2], 'k', label='$p_z$'); ax[1].plot(kk, H['xref'][:, 2], 'C3--', label='ref'); ax[1].set_ylabel('[m]'); ax[1].legend(ncol=2)
for li in range(4): ax[2].plot(kk[:-1], U[:, 3 * li + 2], f'C{li}', drawstyle='steps-post')
ax[2].set_ylabel('$f_z$ [N]'); ax[2].set_xlabel('prediction time [s]')
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'horizon.pdf')); plt.close(fig)

# ---------------- contact sheet png -> report
import shutil; shutil.copy(os.path.join(MEDIA, 'contact_sheet.png'), os.path.join(FIG, 'contact_sheet.png'))
print("figures written")
