# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""Gait scheduling, foothold planning and swing trajectories (all world frame)."""
import numpy as np
from robot import R_from_euler, FOOT_R

class Gait:
    """Periodic contact schedule. Trot: pair A=(LF,RH) then pair B=(RF,LH). duty = stance fraction."""
    def __init__(self, T=0.6, duty=0.5, offsets=(0.0, 0.5, 0.5, 0.0)):
        self.T, self.duty, self.off = T, duty, np.array(offsets)
        self.standing = True; self.t0 = 0.0
    def start(self, t): self.standing = False; self.t0 = t
    def phase(self, t):  # per-leg phase in [0,1)
        return ((t - self.t0) / self.T + self.off) % 1.0
    def contact(self, t):
        if self.standing: return np.ones(4)
        return (self.phase(t) < self.duty).astype(float)
    def settled(self, t, t_settle=0.04, t_pre=0.0):
        """measurement gate: in stance and at least t_settle after touchdown (and t_pre before lift-off)"""
        if self.standing: return np.ones(4)
        ph = self.phase(t) * self.T
        return ((ph >= t_settle) & (ph < self.duty * self.T - t_pre)).astype(float)
    def swing_phase(self, t):
        """sigma in [0,1] within the current swing (0 if in stance); and swing duration"""
        ph = self.phase(t); sg = np.clip((ph - self.duty) / (1 - self.duty), 0, 1)
        return sg, (1 - self.duty) * self.T
    def next_touchdown(self, t, li):
        ph = self.phase(t)[li]
        return t + (1.0 - ph) * self.T if ph >= self.duty else t + (1.0 - ph) * self.T   # next stance start after current swing
    def stance_duration(self): return self.duty * self.T

class Planner:
    """Raibert-heuristic footholds + smooth swing trajectories + horizon references/parameters."""
    def __init__(self, gait, feet_nom, z_com=0.46, h_swing=0.08, k_raibert=0.1, z_land=-0.015):
        self.g = gait; self.feet_nom = np.array(feet_nom)  # nominal foot offsets from COM in body frame (x,y)
        self.z_com, self.h_sw, self.kr, self.z_land = z_com, h_swing, k_raibert, z_land
        self.r_reg = None            # registered (leg-odometry) foot positions during stance
        self.p_lo = None             # lift-off positions
        self.s_prev = np.ones(4)
    def foothold(self, p_com, yaw, v_cmd_b, om_cmd, v_est, li, t_td, t_now):
        """Raibert: hip projection at touchdown + half-stance velocity offset + velocity-error correction.
        v_cmd_b is the body-frame velocity command (forward, lateral, 0)."""
        yaw_td = yaw + om_cmd * (t_td - t_now)
        Rz = R_from_euler([0, 0, yaw_td]); v_cmd = Rz @ v_cmd_b
        hip = p_com + v_est * (t_td - t_now) + Rz @ np.array([self.feet_nom[li, 0], self.feet_nom[li, 1], 0])
        Ts = self.g.stance_duration()
        target = hip + 0.5 * Ts * v_est + self.kr * (v_est - v_cmd)     # Raibert: neutral point from actual velocity + feedback term
        target[2] = self.z_ground()
        return target
    def z_ground(self):
        st = self.s_prev > 0.5
        return float(self.r_reg[st, 2].mean()) if (self.r_reg is not None and st.any()) else 0.0
    def update_contacts(self, t, x_est, p_feet_kin, detected=None):
        """register touchdown / lift-off events from the schedule and from detected contacts (leg odometry anchoring)"""
        s = self.g.contact(t)
        if detected is None: detected = s > 0.5
        if not hasattr(self, 'det_prev'): self.det_prev = detected.copy()
        R = R_from_euler(x_est[3:6]); p = x_est[0:3]
        if self.r_reg is None:
            self.r_reg = np.array([p + R @ pr for pr in p_feet_kin])
            self.p_lo = self.r_reg.copy()
        sm = self.g.settled(t)
        if not hasattr(self, 'sm_prev'): self.sm_prev = sm.copy()
        for li in range(4):
            if (s[li] > 0.5 and self.s_prev[li] < 0.5) or (detected[li] and not self.det_prev[li]):
                self.r_reg[li] = p + R @ p_feet_kin[li]     # scheduled or detected touchdown: provisional anchor
            if sm[li] > 0.5 and self.sm_prev[li] < 0.5:    # settled: final anchor (leg odometry, used by the outputs)
                self.r_reg[li] = p + R @ p_feet_kin[li]
            if s[li] < 0.5 and self.s_prev[li] > 0.5:      # lift-off
                self.p_lo[li] = p + R @ p_feet_kin[li]
        self.s_prev = s.copy(); self.sm_prev = sm.copy(); self.det_prev = detected.copy()
        return s
    def swing_targets(self, t, x_est, v_cmd, om_cmd):
        """desired swing-foot (sphere centre) position/velocity for each leg"""
        s = self.g.contact(t); sg, Tsw = self.g.swing_phase(t)
        p_des = np.zeros((4, 3)); v_des = np.zeros((4, 3))
        for li in range(4):
            if s[li] > 0.5: continue
            t_td = self.g.next_touchdown(t, li)
            tgt = self.foothold(x_est[0:3], x_est[5], v_cmd, om_cmd, x_est[6:9], li, t_td, t)
            a = sg[li]; sm = 10 * a**3 - 15 * a**4 + 6 * a**5; dsm = (30 * a**2 - 60 * a**3 + 30 * a**4) / Tsw
            p_des[li] = self.p_lo[li] + (tgt - self.p_lo[li]) * sm
            v_des[li] = (tgt - self.p_lo[li]) * dsm
            # bump profile 16 a^2 (1-a)^2: zero velocity at lift-off and touchdown; ends z_land below ground (early contact)
            bump = 16 * a**2 * (1 - a)**2; dbump = 32 * a * (1 - a) * (1 - 2 * a) / Tsw
            p_des[li, 2] = self.z_ground() + self.h_sw * bump + self.z_land * sm; v_des[li, 2] = self.h_sw * dbump + self.z_land * dsm
        return p_des, v_des
    def horizon(self, t, x_est, v_cmd, om_cmd, N, dt, d_hat, mass, g=9.81):
        """references xref[(N+1)x12], uref[Nx12] and stage parameters par[(N+1)x22] = [r(12), s(4), d(6)]"""
        xref = np.zeros((N + 1, 12)); uref = np.zeros((N, 12)); par = np.zeros((N + 1, 22))
        r_cur = self.r_reg.copy()
        # future footholds: walk the schedule leg by leg; a leg's foot position changes at each future touchdown
        r_k = np.zeros((N + 1, 4, 3)); s_k = np.zeros((N + 1, 4))
        for li in range(4):
            r = r_cur[li].copy(); s_prev = self.g.contact(t)[li]
            for k in range(N + 1):
                tk = t + k * dt; s = self.g.contact(tk)[li]
                if s > 0.5 and s_prev < 0.5:
                    r = self.foothold(x_est[0:3], x_est[5], v_cmd, om_cmd, x_est[6:9], li, tk, t)
                s_prev = s; s_k[k, li] = s; r_k[k, li] = r
        p_ref = x_est[0:3].copy()
        for k in range(N + 1):
            tk = k * dt; yaw_k = x_est[5] + om_cmd * tk
            v_w = R_from_euler([0, 0, yaw_k]) @ v_cmd                      # body-frame command -> world frame along the turning arc
            if k > 0: p_ref = p_ref + v_w * dt
            xref[k, 0:3] = p_ref; xref[k, 2] = self.z_ground() + self.z_com
            xref[k, 5] = yaw_k
            xref[k, 6:9] = v_w; xref[k, 11] = om_cmd
            par[k, 0:12] = r_k[k].ravel(); par[k, 12:16] = s_k[k]; par[k, 16:22] = d_hat
            if k < N:
                ns = max(s_k[k].sum(), 1.0)
                for li in range(4):
                    if s_k[k, li] > 0.5: uref[k, 3 * li + 2] = mass * g / ns
        return xref, uref, par
