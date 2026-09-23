# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""ctypes bindings for libanymal_nmpc.so (NMPC solver) and libanymal_nmhe.so (MHE solver)."""
import ctypes as C, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, '..', 'c')
D = C.POINTER(C.c_double)
def _p(a): return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(D)

class NMPC:
    def __init__(self):
        self.lib = C.CDLL(os.path.join(CDIR, 'libanymal_nmpc.so'))
        n, nx, nu, nc, npar = (C.c_int() for _ in range(5))
        self.ram = self.lib.nmpc_dims(C.byref(n), C.byref(nx), C.byref(nu), C.byref(nc), C.byref(npar))
        self.N, self.nx, self.nu, self.nc, self.npar = n.value, nx.value, nu.value, nc.value, npar.value
        self.lib.nmpc_kkt.restype = C.c_double; self.lib.nmpc_max_viol.restype = C.c_double; self.lib.nmpc_cycle_fixed.restype = C.c_double
    def init(self, Qd, Rd, QNd, dx_max, du_max):
        self.lib.nmpc_init(_p(Qd), _p(Rd), _p(QNd), C.c_double(dx_max), C.c_double(du_max))
    def set_opt(self, name, val): self.lib.nmpc_set_opt(name.encode(), C.c_double(val))
    def set_ref(self, xref, uref):
        assert xref.shape == (self.N + 1, self.nx) and uref.shape == (self.N, self.nu)
        self.lib.nmpc_set_ref(_p(xref), _p(uref))
    def set_par(self, par):
        assert par.shape == (self.N + 1, self.npar); self.lib.nmpc_set_par(_p(par))
    def engage(self, x0, uinit, tol=1e-6, maxit=100):
        return self.lib.nmpc_engage(_p(x0), _p(uinit), C.c_double(tol), C.c_int(maxit))
    def step(self, xhat, shift=True):
        u0 = np.zeros(self.nu); st = np.zeros(8)
        self.lib.nmpc_step(_p(xhat), C.c_int(int(shift)), u0.ctypes.data_as(D), st.ctypes.data_as(D))
        return u0, st
    def traj(self):
        X = np.zeros((self.N + 1, self.nx)); U = np.zeros((self.N, self.nu))
        self.lib.nmpc_get_traj(X.ctypes.data_as(D), U.ctypes.data_as(D)); return X, U
    def kkt(self, xhat): return self.lib.nmpc_kkt(_p(xhat))
    def cycle_fixed(self, xhat): return self.lib.nmpc_cycle_fixed(_p(xhat))
    def reset_iterate(self, x0, uinit): self.lib.nmpc_reset_iterate(_p(x0), _p(uinit))
    def max_viol(self): return self.lib.nmpc_max_viol()
    def eval_model(self, x, u, par):
        nx, nu, nc = self.nx, self.nu, self.nc
        f = np.zeros(nx); A = np.zeros((nx, nx)); B = np.zeros((nx, nu)); c = np.zeros(nc); Cx = np.zeros((nc, nx)); Cu = np.zeros((nc, nu))
        self.lib.nmpc_eval_model(_p(x), _p(u), _p(par), *(a.ctypes.data_as(D) for a in (f, A, B, c, Cx, Cu)))
        return f, A, B, c, Cx, Cu
    def rk4(self, x, u, par):
        nx, nu = self.nx, self.nu
        xn = np.zeros(nx); Ad = np.zeros((nx, nx)); Bd = np.zeros((nx, nu))
        self.lib.nmpc_rk4(_p(x), _p(u), _p(par), *(a.ctypes.data_as(D) for a in (xn, Ad, Bd)))
        return xn, Ad, Bd

class NMHE:
    def __init__(self):
        self.lib = C.CDLL(os.path.join(CDIR, 'libanymal_nmhe.so'))
        m, nx, nu, nw, ny, nc, ncm = (C.c_int() for _ in range(7))
        self.ram = self.lib.nmhe_dims(*(C.byref(a) for a in (m, nx, nu, nw, ny, nc, ncm)))
        self.M, self.nx, self.nu, self.nw, self.ny, self.nc, self.ncm = (a.value for a in (m, nx, nu, nw, ny, nc, ncm))
    def init(self, Vd, Wd, La0, xa0, dx_max, dw_max, xlb, xub, feas_target=1e-3):
        self.lib.nmhe_init(_p(Vd), _p(Wd), _p(La0), _p(xa0), _p(dx_max), _p(dw_max), _p(xlb), _p(xub), C.c_double(feas_target))
    def set_opt(self, name, val): self.lib.nmhe_set_opt(name.encode(), C.c_double(val))
    def step(self, u_prev, u_t, y):
        xh = np.zeros(self.nx); wh = np.zeros(self.nw); xf = np.zeros(self.nx); st = np.zeros(8)
        self.lib.nmhe_step(_p(u_prev), _p(u_t), _p(y), *(a.ctypes.data_as(D) for a in (xh, wh, xf, st)))
        return xh, wh, xf, st
    def window(self):
        X = np.zeros((self.M + 1, self.nx)); W = np.zeros((self.M, self.nw))
        n = self.lib.nmhe_get_window(X.ctypes.data_as(D), W.ctypes.data_as(D)); return X[:n + 1], W[:n]
    def eval_model(self, x, u, w):
        nx, nu, nw, ny, nc = self.nx, self.nu, self.nw, self.ny, self.nc
        f = np.zeros(nx); A = np.zeros((nx, nx)); G = np.zeros((nx, nw)); h = np.zeros(ny); Hx = np.zeros((ny, nx))
        c = np.zeros(nc); Cx = np.zeros((nc, nx)); Cw = np.zeros((nc, nw))
        self.lib.nmhe_eval_model(_p(x), _p(u), _p(w), *(a.ctypes.data_as(D) for a in (f, A, G, h, Hx, c, Cx, Cw)))
        return f, A, G, h, Hx, c, Cx, Cw

def fd_jac(fun, z, eps=1e-6):
    z = np.array(z, float); f0 = fun(z); J = np.zeros((f0.size, z.size))
    for i in range(z.size):
        zp = z.copy(); zp[i] += eps; zm = z.copy(); zm[i] -= eps
        J[:, i] = (fun(zp) - fun(zm)) / (2 * eps)
    return J

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    mpc, mhe = NMPC(), NMHE()
    print(f"NMPC solver instance: N={mpc.N} nx={mpc.nx} nu={mpc.nu} nc={mpc.nc} static RAM={mpc.ram/1024:.1f} kB")
    print(f"MHE solver  instance: M={mhe.M} nx={mhe.nx} nu={mhe.nu} nw={mhe.nw} ny={mhe.ny} nc={mhe.nc} ncM={mhe.ncm} static RAM={mhe.ram/1024:.1f} kB")
    worst = 0
    for trial in range(5):
        x = rng.normal(size=12) * [0.3, 0.3, 0.1, 0.2, 0.2, 1.0, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5]; x[2] += 0.45
        u = rng.normal(size=12) * 80; r = rng.normal(size=12) * 0.3; s = rng.integers(0, 2, 4).astype(float); dd = rng.normal(size=6) * 20
        par = np.concatenate([r, s, dd])
        f, A, B, c, Cx, Cu = mpc.eval_model(x, u, par)
        worst = max(worst, abs(A - fd_jac(lambda z: mpc.eval_model(z, u, par)[0], x)).max() / (1 + abs(A).max()))
        worst = max(worst, abs(B - fd_jac(lambda z: mpc.eval_model(x, z, par)[0], u)).max() / (1 + abs(B).max()))
        worst = max(worst, abs(Cx - fd_jac(lambda z: mpc.eval_model(z, u, par)[3], x)).max(), abs(Cu - fd_jac(lambda z: mpc.eval_model(x, z, par)[3], u)).max())
        ue = np.concatenate([u, r, s, s, [1.0]]); xe = np.concatenate([x, dd]); wdd = rng.normal(size=6)
        fe, Ae, G, h, Hx, ce, Cxe, Cwe = mhe.eval_model(xe, ue, wdd)
        assert np.allclose(f, fe[:12]) and np.allclose(fe[12:], wdd), "NMPC/MHE dynamics differ!"
        worst = max(worst, abs(Ae - fd_jac(lambda z: mhe.eval_model(z, ue, wdd)[0], xe)).max() / (1 + abs(Ae).max()))
        worst = max(worst, abs(G - fd_jac(lambda z: mhe.eval_model(xe, ue, z)[0], wdd)).max() / (1 + abs(G).max()))
        worst = max(worst, abs(Hx - fd_jac(lambda z: mhe.eval_model(z, ue, wdd)[3], xe)).max() / (1 + abs(Hx).max()))
        worst = max(worst, abs(Cxe - fd_jac(lambda z: mhe.eval_model(z, ue, wdd)[5], xe)).max(), abs(Cwe - fd_jac(lambda z: mhe.eval_model(xe, ue, z)[5], wdd)).max())
        # RK4 sensitivities vs finite differences on the discrete map
        xn, Ad, Bd = mpc.rk4(x, u, par)
        worst = max(worst, abs(Ad - fd_jac(lambda z: mpc.rk4(z, u, par)[0], x)).max(), abs(Bd - fd_jac(lambda z: mpc.rk4(x, z, par)[0], u)).max() / (1 + abs(Bd).max()))
    print(f"max relative derivative error vs central finite differences (5 random points): {worst:.2e}")
    # symbolic <-> C cross check via sympy lambdify
    import sys; sys.path.insert(0, os.path.join(HERE, '..', 'model')); import anymal_srbd as M
    import tempfile
    fnp, hnp = M.generate(os.path.join(tempfile.gettempdir(), '_anymal_model_check.h'))
    e = 0
    for t in range(5):
        x = rng.normal(size=12); u = rng.normal(size=12) * 50; r = rng.normal(size=12); s = rng.integers(0, 2, 4).astype(float); dd = rng.normal(size=6)
        f = mpc.eval_model(x, u, np.concatenate([r, s, dd]))[0]
        e = max(e, abs(f - np.array(fnp(x, u, r, s, dd), float)).max())
        e = max(e, abs(mhe.eval_model(np.concatenate([x, dd]), np.concatenate([u, r, s, s, [1.0]]), np.zeros(6))[3] - np.array(hnp(x, r, s, 1.0), float)).max())
    print(f"C model vs SymPy reference (5 random points): {e:.2e}")
