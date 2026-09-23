# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""Run all experiments for the report. Usage: MUJOCO_GL=osmesa python3 run_walk.py"""
import os, sys, json, time, numpy as np, mujoco, imageio
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results', 'data'); MEDIA = os.path.join(HERE, '..', 'media')
for _d in (RES, MEDIA): os.makedirs(_d, exist_ok=True)
from framework import Framework

VMAX, T_PUSH, F_PUSH, T_TURN, OM_TURN = 0.4, (3.5, 4.3), 40.0, 5.0, 0.3
vcmd = lambda t: np.array([min(VMAX, max(0.0, VMAX * (t - 1.0) / 1.0)), 0.0, 0.0])
omcmd = lambda t: OM_TURN if t >= T_TURN else 0.0
push = lambda t: np.array([0, F_PUSH, 0]) if T_PUSH[0] <= t < T_PUSH[1] else np.zeros(3)

def summarize(L, name):
    t = L['t']; xt = L['x_true']; xh = L['x_hat']; m = t > 1.5
    e = np.sqrt(((xt - xh)[m]**2).mean(0))
    vb = np.array([(np.array([[np.cos(x[5]), np.sin(x[5])], [-np.sin(x[5]), np.cos(x[5])]]) @ x[6:8]) for x in xt])
    tm, te = L['tmpc'], L['tmhe']
    d = dict(name=name, T=float(t[-1]), ok=bool(t[-1] > 7.9), roll_max=float(abs(xt[m, 3]).max()), pitch_max=float(abs(xt[m, 4]).max()),
             z_min=float(xt[m, 2].min()), z_max=float(xt[m, 2].max()), vfwd_walk=float(vb[200:350, 0].mean()), vfwd_turn=float(vb[600:, 0].mean()) if len(t) > 600 else np.nan,
             yawrate_turn=float(xt[600:, 11].mean()) if len(t) > 600 else np.nan, y_push_end=float(xt[430, 1]) if len(t) > 430 else np.nan, y_end=float(xt[-1, 1]),
             est_rms=e.tolist(), mpc_prep_med=float(np.median(tm[:, 0])), mpc_fb_med=float(np.median(tm[:, 1])), mpc_prep_p99=float(np.percentile(tm[:, 0], 99)), mpc_fb_p99=float(np.percentile(tm[:, 1], 99)),
             mpc_prep_max=float(tm[:, 0].max()), mpc_fb_max=float(tm[:, 1].max()), mpc_u0_med=float(np.median(tm[:, 2])),
             mhe_prep_med=float(np.median(te[:, 0])), mhe_fast_med=float(np.median(te[:, 1])), mhe_fb_med=float(np.median(te[:, 2])), mhe_prep_p99=float(np.percentile(te[:, 0], 99)), mhe_fb_p99=float(np.percentile(te[:, 2], 99)),
             mhe_prep_max=float(te[:, 0].max()), mhe_fb_max=float(te[:, 2].max()),
             kkt_med=float(np.median(L['kkt'])), viol_max=float(L['viol'].max()))
    wh = L['w_hat']
    if len(t) > 450:
        d.update(w_walk_mean=wh[200:350].mean(0).tolist(), w_walk_std=wh[200:350].std(0).tolist(), w_push_mean=wh[380:430].mean(0).tolist(), w_after_mean=wh[460:500].mean(0).tolist())
    return d

def truthfed(fw):
    real = fw.mhe.step
    def fake(u_prev, u_t, y):
        xh, wh, xf, st = real(u_prev, u_t, y); xt = fw.rob.true_state()
        return np.concatenate([xt, np.zeros(6)]), np.zeros(6), np.concatenate([xt, np.zeros(6)]), st
    fw.mhe.step = fake

# ------------------------------------------------------------------ renderer
def make_renderer(fw, w=640, h=400):
    ren = mujoco.Renderer(fw.rob.m, h, w); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.4; cam.azimuth = 135; cam.elevation = -18
    def snap(rob, t, fw):
        cam.lookat[:] = rob.d.qpos[:3] + [0, 0, -0.1]
        ren.update_scene(rob.d, cam)
        scn = ren.scene
        # push arrow
        F = push(t)
        if np.linalg.norm(F) > 0 and scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]; base = rob.d.qpos[:3].copy(); tip = base + F / F_PUSH * 0.6
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 0.2, 0.1, 0.9], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, base - F / F_PUSH * 0.6, base)
            scn.ngeom += 1
        # commanded GRF arrows on stance feet
        pf = rob.foot_pos()
        for li in range(4):
            f = fw.f_cmd[li]
            if np.linalg.norm(f) > 5 and scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array([0.1, 0.9, 0.3, 0.8], dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, pf[li], pf[li] + f / 400.0)
                scn.ngeom += 1
        img = ren.render().copy()
        return img
    return snap

def annotate(frames, times, fw_log):
    from PIL import Image, ImageDraw
    out = []
    for img, t in zip(frames, times):
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        phase = 'stand' if t < 1.0 else ('trot + lateral push 40 N' if T_PUSH[0] <= t < T_PUSH[1] else ('trot + turn 0.3 rad/s' if t >= T_TURN else 'trot 0.4 m/s'))
        dr.rectangle([0, 0, 640, 22], fill=(0, 0, 0)); dr.text((6, 5), f"t = {t:5.2f} s   {phase}   |   SRBD NMPC + NMHE   ANYmal-C / MuJoCo", fill=(255, 255, 255))
        out.append(np.asarray(im))
    return out

if __name__ == '__main__':
    results = {}
    # ---------------- E1: main scenario with rendering
    t0 = time.time()
    fw = Framework(noise=True, dist_comp='force', seed=0)
    snap = make_renderer(fw); frame_times = []
    def snap_t(rob, t, fw_): frame_times.append(t); return snap(rob, t, fw_)
    L, frames = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push, render=dict(every=20, fn=snap_t))   # 25 fps
    print(f"E1 main: {time.time()-t0:.1f}s wall, {len(frames)} frames")
    results['main'] = summarize(L, 'main'); np.savez_compressed(os.path.join(RES, 'main.npz'), **L)
    frames = annotate(frames, frame_times, L)
    imageio.mimsave(os.path.join(MEDIA, 'anymal_walk.gif'), frames, duration=0.04, loop=0)
    imageio.mimsave(os.path.join(MEDIA, 'anymal_walk.mp4'), frames, fps=25, quality=8) if False else None
    print("gif written", os.path.getsize(os.path.join(MEDIA, 'anymal_walk.gif')) / 1e6, "MB")
    # contact sheet
    idx = [int(x) for x in np.linspace(0, len(frames) - 1, 8)]
    sheet = np.concatenate([np.concatenate([frames[i] for i in idx[:4]], 1), np.concatenate([frames[i] for i in idx[4:]], 1)], 0)
    imageio.imwrite(os.path.join(MEDIA, 'contact_sheet.png'), sheet)

    # ---------------- E2: disturbance-compensation ablation
    for dc, name in [(False, 'nocomp'), ('force', 'forcecomp'), (True, 'fullcomp')]:
        fw = Framework(noise=True, dist_comp=dc, seed=0)
        L, _ = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push)
        results[name] = summarize(L, name); np.savez_compressed(os.path.join(RES, f'{name}.npz'), **L)
        print(name, results[name]['ok'], results[name]['roll_max'], results[name]['y_push_end'], results[name]['y_end'])
    # ---------------- E3: estimator ablations (truth-fed control; exact-mode NMHE)
    fw = Framework(noise=True, dist_comp=False, seed=0); truthfed(fw)
    L, _ = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push)
    results['truthfed'] = summarize(L, 'truthfed'); np.savez_compressed(os.path.join(RES, 'truthfed.npz'), **L)
    for mode in ['full']:
        fw = Framework(noise=True, dist_comp='force', seed=0, est_mode=mode)
        L, _ = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push)
        results['mhe_' + mode] = summarize(L, 'mhe_' + mode); np.savez_compressed(os.path.join(RES, f'mhe_{mode}.npz'), **L)
    # estimator-only comparison on identical plant trajectory (truth-fed control): reuse vs exact
    for mode in ['fast', 'full']:
        fw = Framework(noise=True, dist_comp=False, seed=0, est_mode=mode)
        real = fw.mhe.step; rec = []
        def fake(u_prev, u_t, y, real=real, fw=fw, rec=rec):
            xh, wh, xf, st = real(u_prev, u_t, y); xt = fw.rob.true_state(); rec.append((xt, xh[:12], xh[12:], st, xf[:12]))
            return np.concatenate([xt, np.zeros(6)]), np.zeros(6), np.concatenate([xt, np.zeros(6)]), st
        fw.mhe.step = fake
        L, _ = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push)
        xt = np.array([r[0] for r in rec]); xh = np.array([r[1] for r in rec]); wh = np.array([r[2] for r in rec]); st = np.array([r[3] for r in rec]); xf = np.array([r[4] for r in rec])
        m = np.arange(len(xt)) > 150
        results['estonly_' + mode] = dict(est_rms=np.sqrt(((xt - xh)[m]**2).mean(0)).tolist(), fast_rms=np.sqrt(((xt - xf)[m]**2).mean(0)).tolist(),
                                          prep_med=float(np.median(st[:, 0])), fb_med=float(np.median(st[:, 2])), fast_med=float(np.median(st[:, 1])), prep_p99=float(np.percentile(st[:, 0], 99)),
                                          w_push=wh[380:430].mean(0).tolist(), w_walk_std=wh[200:350].std(0).tolist())
        np.savez_compressed(os.path.join(RES, f'estonly_{mode}.npz'), xt=xt, xh=xh, wh=wh, st=st[:, :3], xf=xf, t=L['t'], dist_true=L['dist_true'])
        print('estonly', mode, results['estonly_' + mode]['est_rms'][6:9])
    # ---------------- E4: seeds
    seeds = {}
    for seed in range(5):
        fw = Framework(noise=True, dist_comp='force', seed=seed)
        L, _ = fw.run(T_end=8.0, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push)
        seeds[seed] = summarize(L, f'seed{seed}'); print('seed', seed, seeds[seed]['ok'], seeds[seed]['roll_max'])
    results['seeds'] = seeds
    # ---------------- E5: NMPC solver fixed-parameter convergence (cold start at a walking instant)
    fw = Framework(noise=True, dist_comp='force', seed=0)
    L, _ = fw.run(T_end=2.35, t_walk=1.0, v_cmd_fn=vcmd)
    xhat = fw.x_hat.copy(); t = 2.35
    xref, uref, par = fw.plan.horizon(t, xhat, vcmd(t), 0.0, fw.N, fw.dt_c, fw.d_hat, fw.mass)
    fw.mpc.set_ref(xref, uref); fw.mpc.set_par(par)
    conv = {}
    for label, uinit in [('cold (u=0)', np.zeros_like(uref)), ('warm (u=uref)', uref)]:
        fw.mpc.reset_iterate(xhat, uinit); res = []
        for it in range(60):
            res.append(fw.mpc.cycle_fixed(xhat))
            if res[-1] < 1e-9: break
        conv[label] = res; print('convergence', label, len(res), res[-1])
    results['convergence'] = conv
    Xp, Up = fw.mpc.traj(); np.savez_compressed(os.path.join(RES, 'horizon_snapshot.npz'), X=Xp, U=Up, xref=xref, uref=uref, par=par, xhat=xhat)
    json.dump(results, open(os.path.join(RES, 'results.json'), 'w'), indent=1)
    print("done in", time.time() - t0)
