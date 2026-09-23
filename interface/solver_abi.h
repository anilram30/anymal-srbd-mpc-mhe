/*
 * Copyright 2026 Sreeram Anil
 * SPDX-License-Identifier: Apache-2.0
 *
 * Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
 * by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
 * If you use this file, this attribution to Sreeram Anil must be retained.
 */
/* =============================================================================
 *  solver_abi.h -- the C ABI that this simulation framework binds to.
 *
 *  The framework is solver-agnostic: sim/solvers.py loads two shared libraries,
 *
 *      libanymal_nmpc.so   real-time-iteration NMPC for the SRBD model
 *      libanymal_nmhe.so   real-time-iteration MHE  for the augmented SRBD model
 *
 *  and talks to them only through the functions declared below. Each library is
 *  one translation unit: it fixes the problem dimensions at compile time, embeds
 *  the generated model header (c/anymal_models.h, produced by
 *  model/anymal_srbd.py) and wraps a real-time-iteration solver kernel.
 *
 *  WHAT IS NOT IN THIS REPOSITORY
 *  ------------------------------
 *  The two solver kernels themselves, and the wrapper translation units that
 *  embed them, are part of ongoing, unpublished research (a constrained
 *  real-time-iteration NMPC solver and its moving-horizon estimation variant,
 *  developed as an extension of my master's thesis on SLQP-MPC). Their
 *  algorithms and source are confidential until that work is published, so
 *  only the interface is documented here. Everything on the framework side of
 *  this boundary -- model generation, gait scheduling, foothold planning, the
 *  measurement model, the plant, the experiments and the figures -- is public.
 *
 *  ADAPTING A DIFFERENT SOLVER
 *  ---------------------------
 *  Any NMPC/MHE package that offers an RTI-style prepare/feedback split and
 *  handles nonlinear stage inequality constraints can be dropped in behind this
 *  ABI (acados, for instance) without touching a line of Python. See
 *  interface/README.md for the mapping and the conventions each call must obey.
 *
 *  Conventions
 *    - all arrays are row-major, double precision, caller-allocated;
 *    - all times returned in the stats vectors are wall-clock microseconds;
 *    - NX/NU/... are compile-time constants of the library, queried by *_dims;
 *    - no call allocates, and no call is re-entrant.
 * ========================================================================== */
#ifndef SOLVER_ABI_H
#define SOLVER_ABI_H

/* ------------------------------------------------------------------ NMPC ---
 * Problem (see report Sec. IV):  N stages of length TS, state nx = 12,
 * input nu = 12 (four 3-D ground reaction forces), nc stage inequality rows
 * (friction pyramids, normal-force bounds, attitude bounds), and npar stage
 * parameters per stage (foot positions, contact flags, disturbance wrench).
 */

/* Query dimensions; returns the library's static memory footprint in bytes. */
int    nmpc_dims(int* N, int* nx, int* nu, int* nc, int* npar);

/* Set the diagonal stage costs and the per-cycle step bounds. Call once. */
void   nmpc_init(const double* Q, const double* R, const double* QN,
                 double dx_max, double du_max);

/* Scalar options by name. The wrapper accepts the following generic names and
 * maps them onto whatever the underlying kernel calls them:
 *     "penalty"       initial constraint-penalty scale
 *     "penalty_max"   upper bound for the penalty
 *     "feas_target"   target stage-constraint violation
 *     "relin_full"    1 = re-linearise every stage every sample
 *     "relin_tol"     tolerance of the automatic re-linearisation monitor
 *     "cons_hyst"     hysteresis band for constraint activation
 *     "max_inner"     cap on inner iterations per cycle
 * Unknown names must be ignored, not rejected. */
void   nmpc_set_opt(const char* name, double value);

/* References for the current sample: xref is (N+1) x nx, uref is N x nu. */
void   nmpc_set_ref(const double* xref, const double* uref);

/* Stage parameters for the current sample: (N+1) x npar. Stage k of the
 * prediction must be linearised with row k; the wrapper is responsible for
 * routing each row to the model evaluation of its own stage. */
void   nmpc_set_par(const double* par);

/* Initial convergence at a frozen state (cold start from uinit, N x nu).
 * Iterates until the KKT residual falls below tol; returns the cycle count. */
int    nmpc_engage(const double* x0, const double* uinit, double tol, int maxit);

/* One real-time-iteration cycle. shift != 0 advances the warm start by one
 * stage before re-linearising. Writes the input to apply now into u0 and
 * eight diagnostics into stats; the first three entries are required and are
 * the only ones the framework plots:
 *     stats[0] preparation time [us]
 *     stats[1] feedback time    [us]
 *     stats[2] latency from x0 becoming available to u0 being ready [us]
 * entries 3..7 are free for solver-specific counters. */
void   nmpc_step(const double* xhat, int shift, double* u0, double* stats);

/* Current primal trajectory: X is (N+1) x nx, U is N x nu. */
void   nmpc_get_traj(double* X, double* U);

/* KKT residual (inf-norm) and worst stage-constraint violation of the
 * committed iterate, both in the scaling of the problem as posed. */
double nmpc_kkt(const double* xhat);
double nmpc_max_viol(void);

/* Convergence studies: one cycle without shifting (frozen parameters and
 * state), and a reset of the iterate to (x0 repeated, uinit). */
double nmpc_cycle_fixed(const double* xhat);
void   nmpc_reset_iterate(const double* x0, const double* uinit);

/* Direct access to the generated model, used by the verification script
 * (sim/solvers.py) to check the derivatives against finite differences. */
void   nmpc_eval_model(const double* x, const double* u, const double* par,
                       double* f, double* A, double* B,
                       double* c, double* Cx, double* Cu);
void   nmpc_rk4(const double* x, const double* u, const double* par,
                double* xnext, double* Ad, double* Bd);

/* ------------------------------------------------------------------ NMHE ---
 * Problem (see report Sec. V): window of M intervals of length TS, augmented
 * state nx = 18 (12 rigid-body states + 6 disturbance-wrench states), nw = 6
 * estimated inputs (the wrench increments), ny = 21 outputs, nu = 33 known
 * inputs per interval/node (measured contact forces, foot anchors, contact and
 * measurement gates), nc / ncM stage and newest-node inequality rows.
 */
int    nmhe_dims(int* M, int* nx, int* nu, int* nw, int* ny, int* nc, int* ncM);

/* Weights and prior: diagonal output weight V (ny), increment weight W (nw),
 * initial arrival information La0 (nx) about xa0 (nx), per-cycle step bounds
 * dx_max (nx) / dw_max (nw), and a box on the state used as a model domain. */
void   nmhe_init(const double* V, const double* W, const double* La0,
                 const double* xa0, const double* dx_max, const double* dw_max,
                 const double* x_lb, const double* x_ub, double feas_target);

/* Same contract as nmpc_set_opt; the estimator additionally accepts
 * "refresh_tol" (tolerance of the stage-refresh monitor). */
void   nmhe_set_opt(const char* name, double value);

/* One cycle. u_prev is the known input of the interval that has just elapsed,
 * u_t the known input at the new node, y the measurement (ny).
 * Returns the smoothed newest-node estimate in xhat (nx), the newest estimated
 * increment in what (nw), the measurement-innovation estimate available before
 * the inner iterations in xhat_fast (nx), and eight diagnostics in stats with
 * the same first-three convention as nmpc_step, stats[1] being the time to
 * xhat_fast and stats[2] the time to xhat. */
void   nmhe_step(const double* u_prev, const double* u_t, const double* y,
                 double* xhat, double* what, double* xhat_fast, double* stats);

/* Current window: X is (n+1) x nx, W is n x nw, with n = min(samples, M). */
int    nmhe_get_window(double* X, double* W);

/* Oldest-node (arrival) estimate and its information, for diagnostics. */
void   nmhe_get_arrival(double* xa, double* La_diag);

/* Model access for the derivative checks. */
void   nmhe_eval_model(const double* x, const double* u, const double* w,
                       double* f, double* A, double* G, double* h, double* Hx,
                       double* c, double* Cx, double* Cw);

#endif /* SOLVER_ABI_H */
