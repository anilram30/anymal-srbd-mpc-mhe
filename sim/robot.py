# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""ANYmal-C plant in MuJoCo: ground-truth extraction, sensor emulation and low-level leg control."""
import os, json, numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
NOM = json.load(open(os.path.join(HERE, '..', 'model', 'nominal.json')))
LEGS = ['LF', 'RF', 'LH', 'RH']
FOOT_R = NOM['foot_r']

def euler_zyx_from_R(R):
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])
def R_from_euler(e):
    cph, sph = np.cos(e[0]), np.sin(e[0]); cth, sth = np.cos(e[1]), np.sin(e[1]); cps, sps = np.cos(e[2]), np.sin(e[2])
    Rz = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]]); Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]]); Rx = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
    return Rz @ Ry @ Rx

class Robot:
    def __init__(self, xml=os.path.join(HERE, 'scene_torque.xml'), seed=0):
        self.m = mujoco.MjModel.from_xml_path(xml); self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.foot_geom = [NOM['foot_geom'][l] for l in LEGS]
        self.base = self.m.body('base').id
        self.com_off = np.array(NOM['com_off'])          # COM offset from base origin (body frame, nominal)
        self.mass = NOM['mass']
        self.rng = np.random.default_rng(seed); self.noise_scale = 1.0
        self.jacp = np.zeros((3, self.m.nv)); self.jacr = np.zeros((3, self.m.nv))
        self.reset()

    def reset(self):
        d, m = self.d, self.m
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0; d.qpos[2] = NOM['zb']; d.qpos[3] = 1.0
        for li, l in enumerate(LEGS):
            sgn = 1 if l[1] == 'F' else -1
            d.qpos[7 + 3 * li + 1] = sgn * NOM['hfe']; d.qpos[7 + 3 * li + 2] = -sgn * NOM['kfe']
        mujoco.mj_forward(m, d)
        self.q_nom = d.qpos[7:].copy()
        # settle on the ground under a joint-space PD hold (feet load, soft contacts compress)
        for _ in range(400):
            d.ctrl[:] = 200 * (self.q_nom - d.qpos[7:]) - 5 * d.qvel[6:]; mujoco.mj_step(m, d)
        d.ctrl[:] = 0; d.qvel[:] = 0; mujoco.mj_forward(m, d)

    # ------------------------------------------------------------ truth
    def R(self):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, self.d.qpos[3:7]); return R.reshape(3, 3)
    def true_state(self):
        """SRBD state x = [p_com(world), euler ZYX, v_com(world), omega(body)]"""
        d = self.d; mujoco.mj_subtreeVel(self.m, d)
        R = self.R()
        return np.concatenate([d.subtree_com[self.base], euler_zyx_from_R(R), d.subtree_linvel[self.base], d.qvel[3:6]])
    def foot_pos(self):
        """world-frame contact-point positions of the four feet (sphere centre minus radius)"""
        return np.array([self.d.geom_xpos[g] - [0, 0, FOOT_R] for g in self.foot_geom])
    def foot_vel(self):
        v = []
        for li, g in enumerate(self.foot_geom):
            mujoco.mj_jacGeom(self.m, self.d, self.jacp, self.jacr, g); v.append(self.jacp @ self.d.qvel)
        return np.array(v)
    def leg_jac(self, li, contact_point=False):
        """world-frame 3x3 foot Jacobian wrt the leg's 3 joints, and 3x3 wrt base angular velocity.
        contact_point=True evaluates the Jacobian at the ground contact point (sphere centre minus radius),
        which is the point that is stationary in stance (the sphere may roll)."""
        if contact_point:
            pt = self.d.geom_xpos[self.foot_geom[li]] - np.array([0, 0, FOOT_R])
            mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, pt, self.m.geom_bodyid[self.foot_geom[li]])
        else:
            mujoco.mj_jacGeom(self.m, self.d, self.jacp, self.jacr, self.foot_geom[li])
        return self.jacp[:, 6 + 3 * li:9 + 3 * li].copy(), self.jacp[:, 3:6].copy()
    def contact_forces(self):
        """true ground reaction force on each foot (world frame) from MuJoCo contacts"""
        f = np.zeros((4, 3)); d, m = self.d, self.m; wr = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for li, g in enumerate(self.foot_geom):
                if c.geom1 == g or c.geom2 == g:
                    mujoco.mj_contactForce(m, d, i, wr)
                    fr = c.frame.reshape(3, 3)          # rows: normal, tangent1, tangent2
                    fw = fr.T @ wr[:3]
                    if c.geom1 == g: fw = -fw           # force on geom2 by geom1 convention
                    f[li] += fw
        return f

    # ------------------------------------------------------------ sensors (NMHE)
    def measure(self, s_meas, noise=True):
        """Emulated proprioception at the current time. s_meas: settled-stance gates.
        returns y = [euler(3), gyro(3), g_v * v_com_kin(3), sm_i * p_rel_i (12, body frame, COM-relative)] and
        the raw kinematic quantities used by the leg-odometry registration."""
        s_contact = s_meas
        d = self.d; R = self.R()
        n = (lambda k, sd: self.rng.normal(size=k) * sd * self.noise_scale) if noise else (lambda k, sd: np.zeros(k))
        eul = euler_zyx_from_R(R) + n(3, 0.005)
        om = d.qvel[3:6] + n(3, 0.02)
        qd = d.qvel[6:] + n(12, 0.05)
        pb = d.qpos[:3]
        com_off = R.T @ (d.subtree_com[self.base] - pb)         # kinematic COM offset (function of q only)
        mujoco.mj_subtreeVel(self.m, d)
        v_com_rel = d.subtree_linvel[self.base] - d.qvel[0:3]   # COM velocity relative to the base (joint motion, kinematic)
        prel = np.zeros((4, 3)); vk = []; 
        for li in range(4):
            Jl, Jr = self.leg_jac(li, contact_point=True)
            pf = d.geom_xpos[self.foot_geom[li]] - [0, 0, FOOT_R]
            prel[li] = R.T @ (pf - pb) - com_off + n(3, 0.003)   # FK in body frame, COM-relative
            if s_contact[li] > 0.5:
                vb = -(Jr @ om + Jl @ qd[3 * li:3 * li + 3])            # base velocity from a stationary foot
                vk.append(vb + v_com_rel)
        vcom = np.mean(vk, axis=0) if vk else np.zeros(3)
        y = np.concatenate([eul, om, vcom, (prel * np.array(s_meas)[:, None]).ravel()])
        return y, prel

    # ------------------------------------------------------------ actuation (500 Hz)
    def apply(self, f_cmd, s_contact, p_sw, v_sw, kp=np.array([900., 900., 900.]), kd=np.array([30., 30., 30.])):
        """stance legs: tau = -J^T f (map commanded GRF to joint torques);
        swing legs: Cartesian PD on the foot + gravity/bias compensation."""
        d = self.d; tau = np.zeros(12)
        pf = self.foot_pos(); vf = self.foot_vel()
        for li in range(4):
            Jl, _ = self.leg_jac(li)
            if s_contact[li] > 0.5:   # gravity/bias-compensated GRF mapping: realised foot force == commanded
                tau[3 * li:3 * li + 3] = -Jl.T @ f_cmd[li] + d.qfrc_bias[6 + 3 * li:9 + 3 * li]
            else:
                F = kp * (p_sw[li] - pf[li]) + kd * (v_sw[li] - vf[li])
                tau[3 * li:3 * li + 3] = Jl.T @ F + d.qfrc_bias[6 + 3 * li:9 + 3 * li]
        d.ctrl[:] = np.clip(tau, -80, 80)
        return tau
    def push(self, F_world):
        self.d.xfrc_applied[self.base, :3] = F_world
    def step(self):
        mujoco.mj_step(self.m, self.d)
