# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""
ANYmal-C single-rigid-body (SRBD) model: single symbolic source of truth.

Emits ../c/anymal_models.h with two function sets, both consumed *unmodified*
by Ram's solvers:

  NMPC solver:   anymal_fc/fx/fu(x,u,o), anymal_c/cx/cu(x,u,o), anymal_cN/cNx(x,o)
                   stage parameters (foot positions r_i, contact flags s_i, disturbance d)
                   are read from a per-stage table resolved by the pointer offset of u
                   inside the static solver input array (see anymal_stage()).
  MHE solver:   anymalmhe_fc/fcx/fcw(x,u,w,o), anymalmhe_h/hx(x,u,o),
                   anymalmhe_c/cx/cw(x,w,o), anymalmhe_cM/cMx(x,o)
                   stage parameters travel inside the *known input* u = [f, r, s, s_meas, g_v].

State  x = [p(3) world, th(3) ZYX Euler (roll,pitch,yaw), v(3) world, om(3) body]
NMPC u = f = [f_LF, f_RF, f_LH, f_RH] world-frame ground reaction forces (12)
NMHE state x_e = [x, d] with d = [d_f (3, world force), d_tau (3, body torque)] the unknown wrench
     modelled as a random walk d_dot = w (nw = 6, the estimated disturbance input)
"""
import json, os, sys
import sympy as sp
from sympy.printing.c import C99CodePrinter

HERE = os.path.dirname(os.path.abspath(__file__))
NOM = json.load(open(os.path.join(HERE, 'nominal.json')))
MASS = NOM['mass']; ICOM = sp.Matrix(NOM['Icom']); GRAV = 9.81
MU = 0.6            # friction coefficient used by the controller (MuJoCo default 1.0 -> margin)
FMAX = 400.0        # per-foot vertical force bound [N]
ANG_MAX = 0.35      # roll/pitch bound [rad] (NMPC)
ANG_MAX_E = 0.6     # roll/pitch bound [rad] (NMHE)
DF_MAX, DT_MAX = 800.0, 300.0   # disturbance wrench bounds for the estimator
NLEG = 4

# ----------------------------------------------------------------- symbols
p = sp.Matrix(sp.symbols('px py pz'))
th = sp.Matrix(sp.symbols('phi theta psi'))
v = sp.Matrix(sp.symbols('vx vy vz'))
om = sp.Matrix(sp.symbols('wx wy wz'))
f = sp.Matrix(sp.symbols(' '.join(f'f{i}{a}' for i in range(NLEG) for a in 'xyz')))
r = sp.Matrix(sp.symbols(' '.join(f'r{i}{a}' for i in range(NLEG) for a in 'xyz')))
s = sp.Matrix(sp.symbols('s0 s1 s2 s3'))
sm = sp.Matrix(sp.symbols('sm0 sm1 sm2 sm3'))   # measurement gates (settled stance)
gv = sp.Symbol('gv')                             # kinematic-velocity gate
d = sp.Matrix(sp.symbols('dfx dfy dfz dtx dty dtz'))
x = sp.Matrix.vstack(p, th, v, om)

def Rz(a): return sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
def Ry(a): return sp.Matrix([[sp.cos(a), 0, sp.sin(a)], [0, 1, 0], [-sp.sin(a), 0, sp.cos(a)]])
def Rx(a): return sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
def skew(a): return sp.Matrix([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

phi, theta, psi = th
R = Rz(psi) * Ry(theta) * Rx(phi)                      # body -> world
# Euler-rate map: th_dot = T(th) * om_body   (ZYX convention)
T = sp.Matrix([[1, sp.sin(phi) * sp.tan(theta), sp.cos(phi) * sp.tan(theta)],
               [0, sp.cos(phi), -sp.sin(phi)],
               [0, sp.sin(phi) / sp.cos(theta), sp.cos(phi) / sp.cos(theta)]])

def srbd_xdot(fvec, rvec, svec, dvec):
    Fsum = sp.zeros(3, 1); Tsum = sp.zeros(3, 1)
    for i in range(NLEG):
        fi = svec[i] * fvec[3 * i:3 * i + 3, 0]
        ri = rvec[3 * i:3 * i + 3, 0]
        Fsum += fi
        Tsum += skew(ri - p) * fi                          # world-frame moment about COM
    dF = dvec[0:3, 0]; dT = dvec[3:6, 0]
    pdot = v
    thdot = T * om
    vdot = Fsum / MASS + sp.Matrix([0, 0, -GRAV]) + dF / MASS
    omdot = ICOM.inv() * (R.T * Tsum + dT - skew(om) * (ICOM * om))
    return sp.Matrix.vstack(pdot, thdot, vdot, omdot)

# ----------------------------------------------------------- constraints
def cone_rows(fvec):
    rows = []
    for i in range(NLEG):
        fx, fy, fz = fvec[3 * i:3 * i + 3, 0]
        rows += [(fz - FMAX) / FMAX, -fz / FMAX,
                 (fx - MU * fz) / FMAX, (-fx - MU * fz) / FMAX,
                 (fy - MU * fz) / FMAX, (-fy - MU * fz) / FMAX]
    return rows

def angle_rows(amax):
    return [(phi - amax) / amax, (-phi - amax) / amax, (theta - amax) / amax, (-theta - amax) / amax]

# NMPC: 24 cone/bound rows + 4 attitude rows = 28 stage rows; no terminal rows
c_nmpc = cone_rows(f) + angle_rows(ANG_MAX)
# NMHE: augmented state x_e = [x (12), d (6)] with random-walk disturbance d_dot = w (nw = 6).
# stage rows: 12 wrench bounds (state rows) + 4 attitude rows + 6 rate-of-change bounds on w; newest node: 12 + 4 rows
DDOT_MAX = 2000.0   # |d_dot| bound [N/s, Nm/s]
wd = sp.Matrix(sp.symbols('wd0 wd1 wd2 wd3 wd4 wd5'))
x_e = sp.Matrix.vstack(x, d)
wrench_rows = ([(d[i] - DF_MAX) / DF_MAX for i in range(3)] + [(-d[i] - DF_MAX) / DF_MAX for i in range(3)]
               + [(d[i] - DT_MAX) / DT_MAX for i in range(3, 6)] + [(-d[i] - DT_MAX) / DT_MAX for i in range(3, 6)])
c_nmhe = wrench_rows + angle_rows(ANG_MAX_E) + [(wd[i] - DDOT_MAX) / DDOT_MAX for i in range(6)] + [(-wd[i] - DDOT_MAX) / DDOT_MAX for i in range(6)]
cM_nmhe = wrench_rows + angle_rows(ANG_MAX_E)

# NMHE outputs: y = [th (IMU attitude), om (gyro), v (leg-odometry velocity),
#                    s_i * R^T (r_i - p)  (leg kinematics, per foot, body frame)]  -> ny = 21
def h_nmhe(rvec, smvec, g):
    rows = list(th) + list(om) + list(g * v)
    for i in range(NLEG):
        rel = R.T * (rvec[3 * i:3 * i + 3, 0] - p)
        rows += list(smvec[i] * rel)
    return rows

# ---------------------------------------------------------------- emitters
pr = C99CodePrinter()

def emit(name, args, exprs, prologue=()):
    """args: list of (cname, [symbols]) ; prologue: extra C lines after arg unpack"""
    exprs = [sp.sympify(e) for e in exprs]
    used = set().union(*[e.free_symbols for e in exprs]) if exprs else set()
    lines = [f"static inline void {name}({', '.join(f'const double* {a}' for a, _ in args)}, double* o) {{"]
    for a, syms in args:
        hit = False
        for i, sy in enumerate(syms):
            if sy in used:
                lines.append(f"    const double {sy} = {a}[{i}];"); hit = True
        if not hit: lines.append(f"    (void){a};")
    lines += list(prologue)
    reps, red = sp.cse(exprs, symbols=sp.numbered_symbols('t'), optimizations='basic')
    for lhs, rhs in reps: lines.append(f"    const double {lhs} = {pr.doprint(rhs)};")
    for i, e in enumerate(red): lines.append(f"    o[{i}] = {pr.doprint(e)};")
    if not red: lines.append("    (void)o;")
    lines.append("}\n")
    return '\n'.join(lines)

def emit_par(name, args, exprs):
    """NMPC solver variant: parameters (r,s,d) come from the stage table via u's pointer offset."""
    exprs = [sp.sympify(e) for e in exprs]
    used = set().union(*[e.free_symbols for e in exprs]) if exprs else set()
    pro = ["    const double* par = anymal_par_of_u(u);"]
    parsyms = list(r) + list(s) + list(d)
    for i, sy in enumerate(parsyms):
        if sy in used: pro.append(f"    const double {sy} = par[{i}];")
    return emit(name, args, exprs, prologue=pro)

def jac(exprs, syms):
    return list(sp.Matrix(exprs).jacobian(sp.Matrix(syms))) if exprs else []

def generate(path):
    xd_nmpc = srbd_xdot(f, r, s, d)
    u_e = sp.Matrix.vstack(f, r, s, sm, sp.Matrix([gv]))  # MHE solver known input (33)
    xd_nmhe = sp.Matrix.vstack(srbd_xdot(f, r, s, d), wd)   # identical rigid-body expressions + d_dot = w
    h = h_nmhe(r, sm, gv)
    xs, fs, ds = list(x), list(f), list(d)
    xes, wds = list(x_e), list(wd)
    out = ["/* AUTO-GENERATED by model/anymal_srbd.py -- ANYmal-C SRBD model for the RTI NMPC and MHE solvers.",
           f" * mass={MASS:.4f} kg, I_com=diag-ish {[float(ICOM[i,i]) for i in range(3)]}, mu={MU}, fmax={FMAX}",
           " * DO NOT EDIT. */",
           "#ifndef ANYMAL_MODELS_H", "#define ANYMAL_MODELS_H", "#include <math.h>",
           f"#define ANYMAL_NX 12", f"#define ANYMALMHE_NX 18", f"#define ANYMAL_NU {NLEG*3}", f"#define ANYMAL_NPAR {12+4+6}",
           f"#define ANYMAL_NC {len(c_nmpc)}", f"#define ANYMAL_NCN 0",
           f"#define ANYMALMHE_NU {12+12+4+4+1}", f"#define ANYMALMHE_NW 6", f"#define ANYMALMHE_NY {len(h)}",
           f"#define ANYMALMHE_NC {len(c_nmhe)}", f"#define ANYMALMHE_NCM {len(cM_nmhe)}",
           f"#define ANYMAL_MASS {MASS!r}", f"#define ANYMAL_MU {MU!r}", f"#define ANYMAL_FMAX {FMAX!r}",
           "",
           "/* ---- NMPC solver stage-parameter table: filled by the driver every sample.",
           " *  The solver passes u == &S.U[k][0]; the stage index is recovered from the",
           " *  pointer offset (the solver library is unmodified). anymal_par_ubase must point to S.U[0]. */",
           "extern const double* anymal_par_ubase;",
           "extern double anymal_par_table[][ANYMAL_NPAR];",
           "static inline const double* anymal_par_of_u(const double* u) {",
           "    const long k = (long)(u - anymal_par_ubase) / ANYMAL_NU;",
           "    return anymal_par_table[k];",
           "}", ""]
    # ---- NMPC solver set
    out.append(emit_par('anymal_fc', [('x', xs), ('u', fs)], list(xd_nmpc)))
    out.append(emit_par('anymal_fx', [('x', xs), ('u', fs)], jac(list(xd_nmpc), xs)))
    out.append(emit_par('anymal_fu', [('x', xs), ('u', fs)], jac(list(xd_nmpc), fs)))
    out.append(emit_par('anymal_c', [('x', xs), ('u', fs)], c_nmpc))
    out.append(emit_par('anymal_cx', [('x', xs), ('u', fs)], jac(c_nmpc, xs)))
    out.append(emit_par('anymal_cu', [('x', xs), ('u', fs)], jac(c_nmpc, fs)))
    out.append(emit('anymal_cN', [('x', xs)], []))
    out.append(emit('anymal_cNx', [('x', xs)], []))
    # ---- MHE solver set
    ue = list(u_e)
    out.append(emit('anymalmhe_fc', [('x', xes), ('u', ue), ('w', wds)], list(xd_nmhe)))
    out.append(emit('anymalmhe_fcx', [('x', xes), ('u', ue), ('w', wds)], jac(list(xd_nmhe), xes)))
    out.append(emit('anymalmhe_fcw', [('x', xes), ('u', ue), ('w', wds)], jac(list(xd_nmhe), wds)))
    out.append(emit('anymalmhe_h', [('x', xes), ('u', ue)], h))
    out.append(emit('anymalmhe_hx', [('x', xes), ('u', ue)], jac(h, xes)))
    out.append(emit('anymalmhe_c', [('x', xes), ('w', wds)], c_nmhe))
    out.append(emit('anymalmhe_cx', [('x', xes), ('w', wds)], jac(c_nmhe, xes)))
    out.append(emit('anymalmhe_cw', [('x', xes), ('w', wds)], jac(c_nmhe, wds)))
    out.append(emit('anymalmhe_cM', [('x', xes)], cM_nmhe))
    out.append(emit('anymalmhe_cMx', [('x', xes)], jac(cM_nmhe, xes)))
    out.append("#endif /* ANYMAL_MODELS_H */")
    open(path, 'w').write('\n'.join(out))
    # numpy lambdas for verification
    import numpy as np
    fnp = sp.lambdify([xs, fs, list(r), list(s), ds], list(xd_nmpc), 'numpy')
    hnp = sp.lambdify([xs, list(r), list(sm), gv], h, 'numpy')
    return fnp, hnp

if __name__ == '__main__':
    generate(os.path.join(HERE, '..', 'c', 'anymal_models.h'))
    print('wrote c/anymal_models.h')
