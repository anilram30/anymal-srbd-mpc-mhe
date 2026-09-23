# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""Unified SRBD NMPC + NMHE output-feedback walking loop for ANYmal-C in MuJoCo."""
import numpy as np, json, os
from robot import Robot, LEGS, R_from_euler, NOM
from gait import Gait, Planner
from solvers import NMPC, NMHE

class Framework:
    def __init__(self, T_gait=0.7, xml=None, seed=0, noise=True, dist_comp=True, est_mode='fast', ekf_baseline=False, Q=None, R=None):
        self.rob = Robot(seed=seed) if xml is None else Robot(xml=xml, seed=seed); self.noise = noise; self.dist_comp = dist_comp
        self.dt_sim = self.rob.dt                              # 2 ms
        self.mpc, self.mhe = NMPC(), NMHE()
        self.N, self.dt_c = self.mpc.N, 0.01                   # NMPC 50 Hz
        self.M, self.dt_e = self.mhe.M, 0.01                   # NMHE 100 Hz
        self.n_c, self.n_e = int(round(self.dt_c / self.dt_sim)), int(round(self.dt_e / self.dt_sim))
        self.gait = Gait(T=T_gait)
        feet_nom = np.array(NOM['feet_rel_com'])[:, :2]
        self.plan = Planner(self.gait, feet_nom, z_com=0.45, h_swing=0.06, k_raibert=0.12, z_land=-0.015)
        self.mass = self.rob.mass
        # ---- NMPC weights / options
        Q = np.array([5, 5, 300, 100, 100, 50, 20, 20, 20, 1, 1, 2], float) if Q is None else np.array(Q, float)
        R = np.full(12, 1e-4) if R is None else np.full(12, float(R)); QN = 2 * Q
        self.mpc.init(Q, R, QN, dx_max=3.0, du_max=600.0)
        self.mpc.set_opt('penalty', 50.0); self.mpc.set_opt('penalty_max', 1e4); self.mpc.set_opt('feas_target', 1e-3)
        # ---- NMHE weights / options
        V = np.concatenate([np.full(3, 1 / 0.01**2), np.full(3, 1 / 0.05**2), np.full(3, 1 / 0.03**2), np.full(12, 1 / 0.01**2)])
        W = np.concatenate([np.full(3, 1 / 800.0**2), np.full(3, 1 / 200.0**2)])       # random-walk increments d_dot [N/s, Nm/s]
        x0 = np.concatenate([self.rob.true_state(), np.zeros(6)])                     # augmented: [x, d]
        La0 = np.concatenate([np.full(3, 1e2), np.full(3, 1e3), np.full(3, 1e1), np.full(3, 1e1), np.full(3, 1e-3), np.full(3, 1e-2)])
        dxm = np.array([1, 1, 1, .5, .5, 1, 2, 2, 2, 3, 3, 3, 400, 400, 400, 150, 150, 150.]); dwm = np.full(6, 4000.)
        xlb = np.array([-1e3, -1e3, 0.05, -1.2, -1.2, -1e3, -5, -5, -5, -20, -20, -20, -1e3, -1e3, -1e3, -400, -400, -400]); xub = -xlb; xub[2] = 1.5
        self.mhe.init(V, W, La0, x0, dxm, dwm, xlb, xub, feas_target=1e-3)
        self.mhe.set_opt('relin_full', 1.0 if est_mode == 'full' else 0.0)
        self.ekf_baseline = ekf_baseline
        self.log = {k: [] for k in ['t', 'x_true', 'x_hat', 'w_hat', 'f_cmd', 'f_true', 's', 'y', 'tmpc', 'tmhe', 'kkt', 'viol', 'x_fast', 'dist_true', 'q', 'tau', 'u_prev', 'wd_hat']}
        self.d_hat = np.zeros(6); self.f_cmd = np.zeros((4, 3)); self.x_hat = x0.copy(); self.w_hat = np.zeros(6)
        self.u_prev = None; self.engaged = False; self.dist_true = np.zeros(6)
        self.w_hist = []; self.n_ma = int(round(T_gait / self.dt_e))

    def known_input(self, t, s, f=None):
        """NMHE known input u = [f (12), r (12), s (4), s_meas (4), g_v (1)]: realised GRF (torque-sensor based
        reconstruction, emulated from the plant contact forces + noise), registered foot positions, the contact
        schedule, the settled-stance measurement gates and the kinematic-velocity gate"""
        if f is None: f = self.f_meas()
        s_dyn = np.maximum(s, self.s_detect(f))                   # dynamics gate: scheduled OR detected contact
        sm = self.gait.settled(t) * self.s_detect(f)                # output gate: settled scheduled stance AND detected
        return np.concatenate([f.ravel(), self.plan.r_reg.ravel(), s_dyn, sm, [1.0 if sm.sum() > 0 else 0.0]])
    def f_meas(self):
        """realised GRF per foot (torque-sensor reconstruction emulated by the plant contact forces + 2 N noise)"""
        ft = self.rob.contact_forces()
        return ft + (self.rob.rng.normal(size=ft.shape) * 2.0 * self.rob.noise_scale if self.noise else 0.0)
    def s_detect(self, f, thr=8.0):
        return (np.linalg.norm(f, axis=1) > thr).astype(float)

    def run(self, T_end=8.0, v_cmd_fn=lambda t: np.zeros(3), om_cmd_fn=lambda t: 0.0, t_walk=1.0, push_fn=None, render=None):
        rob, mpc, mhe = self.rob, self.mpc, self.mhe
        nsteps = int(round(T_end / self.dt_sim)); frames = []
        p_sw = np.zeros((4, 3)); v_sw = np.zeros((4, 3))
        for i in range(nsteps):
            t = i * self.dt_sim
            if (not self.gait.standing) is False and t >= t_walk: self.gait.start(t)
            v_cmd, om_cmd = v_cmd_fn(t), om_cmd_fn(t)
            # ---------- external disturbance (truth)
            F_ext = push_fn(t) if push_fn else np.zeros(3)
            rob.push(F_ext); self.dist_true = np.concatenate([F_ext, np.zeros(3)])
            # ---------- NMHE at 100 Hz
            if i % self.n_e == 0:
                x_true = rob.true_state()
                f_now = self.f_meas(); det = self.s_detect(f_now) > 0.5
                y, prel = rob.measure(self.gait.settled(t) * det, noise=self.noise)
                s = self.plan.update_contacts(t, self.x_hat, prel, detected=det)   # anchors r_reg (leg odometry)
                u_t = self.known_input(t, s, f=f_now)
                if self.u_prev is None: self.u_prev = u_t.copy()
                self.log['u_prev'].append(self.u_prev.copy())
                xe_hat, wd_hat, xe_fast, st_e = mhe.step(self.u_prev, u_t, y); self.log['wd_hat'].append(wd_hat.copy())
                x_hat, w_hat, x_fast = xe_hat[:12], xe_hat[12:], xe_fast[:12]   # disturbance wrench = augmented state at the newest node
                self.x_hat, self.w_hat = x_hat, w_hat
                if self.dist_comp == 'force': self.d_hat = np.concatenate([w_hat[:3], np.zeros(3)])
                elif self.dist_comp: self.d_hat = w_hat.copy()
                else: self.d_hat = np.zeros(6)
                self.log['t'].append(t); self.log['x_true'].append(x_true); self.log['x_hat'].append(x_hat.copy())
                self.log['w_hat'].append(w_hat.copy()); self.log['x_fast'].append(x_fast.copy()); self.log['y'].append(y)
                self.log['tmhe'].append(st_e); self.log['dist_true'].append(self.dist_true.copy()); self.log['s'].append(s.copy())
                self.log['f_true'].append(rob.contact_forces()); self.log['f_cmd'].append(self.f_cmd.copy())
                self.log['q'].append(rob.d.qpos.copy())
            # ---------- NMPC at 50 Hz
            if i % self.n_c == 0:
                xref, uref, par = self.plan.horizon(t, self.x_hat, v_cmd, om_cmd, self.N, self.dt_c, self.d_hat, self.mass)
                mpc.set_ref(xref, uref); mpc.set_par(par)
                if not self.engaged:
                    it = mpc.engage(self.x_hat, uref, tol=1e-6, maxit=100); self.engaged = True
                    print(f"[engage] NMPC initial convergence in {it} cycles, KKT={mpc.kkt(self.x_hat):.2e}")
                u0, st_c = mpc.step(self.x_hat, shift=True)
                self.f_cmd = u0.reshape(4, 3) * self.gait.contact(t)[:, None]
                self.log['tmpc'].append(st_c); self.log['kkt'].append(mpc.kkt(self.x_hat)); self.log['viol'].append(mpc.max_viol())
            if i % self.n_e == 0:      # start of an NMHE interval: reset the realised-force accumulator
                self.f_acc = np.zeros((4, 3)); self.n_acc = 0
                self.u_prev = self.known_input(t, self.gait.contact(t), f=np.zeros((4, 3)))
            # ---------- low-level 500 Hz
            s = self.gait.contact(t)
            p_sw, v_sw = self.plan.swing_targets(t, self.x_hat, v_cmd, om_cmd)
            tau = rob.apply(self.f_cmd, s, p_sw, v_sw)
            if i % self.n_e == 0: self.log['tau'].append(tau.copy())
            rob.step()
            self.f_acc += self.f_meas(); self.n_acc += 1
            fm = self.f_acc / max(self.n_acc, 1)
            self.u_prev[:12] = fm.ravel()                                        # mean realised GRF over the interval
            self.u_prev[24:28] = np.maximum(self.gait.contact(t), self.s_detect(fm, thr=4.0))
            if render is not None and i % render['every'] == 0:
                frames.append(render['fn'](rob, t, self))
            if not np.isfinite(rob.d.qpos).all() or rob.d.qpos[2] < 0.15:
                print(f"!! robot fell at t={t:.2f}s"); break
        L = {k: np.array(v) for k, v in self.log.items()}
        return L, frames
