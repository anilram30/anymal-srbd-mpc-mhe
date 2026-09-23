# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""High-definition animation of the closed-loop walk.

Usage:  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python3 render_hd.py [--width 1920] [--fps 50]

Writes results/anymal_walk_hd.mp4 (web/H.264), .webm (VP9) and a compact looping GIF.
"""
import os, sys, argparse, numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'media'); os.makedirs(RES, exist_ok=True)
from framework import Framework

VMAX, T_PUSH, F_PUSH, T_TURN, OM_TURN = 0.4, (3.5, 4.3), 40.0, 5.0, 0.3
vcmd = lambda t: np.array([min(VMAX, max(0.0, VMAX * (t - 1.0) / 1.0)), 0.0, 0.0])
omcmd = lambda t: OM_TURN if t >= T_TURN else 0.0
push = lambda t: np.array([0, F_PUSH, 0]) if T_PUSH[0] <= t < T_PUSH[1] else np.zeros(3)

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
GREEN, RED, ORANGE, WHITE = (0.20, 0.85, 0.35, 0.85), (0.95, 0.25, 0.15, 0.92), (1.0, 0.65, 0.10, 0.92), (235, 238, 242)


def arrow(scn, frm, to, rgba, width):
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(np.asarray(to) - np.asarray(frm)) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(frm, float), np.asarray(to, float))
    scn.ngeom += 1


def phase_of(t):
    if t < 1.0:
        return 'standing  ·  controller and estimator engaged'
    if T_PUSH[0] <= t < T_PUSH[1]:
        return 'trot  ·  40 N lateral disturbance applied'
    if t >= T_TURN:
        return 'trot  ·  turning at 0.3 rad/s'
    return 'trot  ·  forward command 0.4 m/s'


def overlay(img, t, hud, W, H):
    s = W / 1920.0
    im = Image.fromarray(img).convert('RGB')
    dr = ImageDraw.Draw(im, 'RGBA')
    fb = ImageFont.truetype(FONTB, int(34 * s)); fm = ImageFont.truetype(FONT, int(25 * s))
    fs = ImageFont.truetype(FONT, int(22 * s)); fsb = ImageFont.truetype(FONTB, int(24 * s))
    # top banner
    dr.rectangle([0, 0, W, int(96 * s)], fill=(8, 12, 18, 205))
    dr.text((int(34 * s), int(16 * s)), 'ANYmal-C  ·  single-rigid-body nonlinear MPC + moving-horizon estimation', font=fb, fill=WHITE)
    dr.text((int(34 * s), int(58 * s)), phase_of(t), font=fm, fill=(150, 200, 235))
    tt = f't = {t:5.2f} s'
    dr.text((W - int(34 * s) - dr.textlength(tt, font=fb), int(28 * s)), tt, font=fb, fill=WHITE)
    # HUD panel
    x0, y0, w, lh = int(34 * s), H - int(232 * s), int(430 * s), int(31 * s)
    dr.rounded_rectangle([x0, y0, x0 + w, H - int(34 * s)], radius=int(10 * s), fill=(8, 12, 18, 190))
    rows = [('forward speed', f"{hud['v']:.2f} / {hud['vc']:.2f} m/s  (est / cmd)"),
            ('roll · pitch', f"{np.degrees(hud['roll']):+.1f}° · {np.degrees(hud['pitch']):+.1f}°"),
            ('yaw rate', f"{hud['wz']:+.2f} / {hud['wzc']:+.2f} rad/s"),
            ('base height', f"{hud['z']:.3f} m"),
            ('est. disturbance', f"{np.linalg.norm(hud['d']):5.1f} N")]
    for i, (k, v) in enumerate(rows):
        y = y0 + int(14 * s) + i * lh
        dr.text((x0 + int(16 * s), y), k, font=fs, fill=(150, 165, 180))
        dr.text((x0 + int(190 * s), y), v, font=fsb, fill=WHITE)
    # legend
    lx, ly = W - int(34 * s) - int(372 * s), H - int(132 * s)
    dr.rounded_rectangle([lx, ly, lx + int(372 * s), H - int(34 * s)], radius=int(10 * s), fill=(8, 12, 18, 175))
    for i, (c, lab) in enumerate([((52, 217, 90), 'commanded contact forces'),
                                  ((242, 64, 38), 'applied disturbance'),
                                  ((255, 166, 26), 'estimated disturbance')]):
        y = ly + int(14 * s) + i * lh
        dr.rectangle([lx + int(16 * s), y + int(8 * s), lx + int(40 * s), y + int(16 * s)], fill=c)
        dr.text((lx + int(54 * s), y), lab, font=fs, fill=WHITE)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--fps', type=int, default=50)
    ap.add_argument('--tend', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    W, H = a.width, int(round(a.width * 9 / 16))
    every = max(1, int(round(1.0 / (a.fps * 0.002))))          # plant steps between frames (dt = 2 ms)
    fps = 1.0 / (every * 0.002)

    fw = Framework(noise=True, dist_comp='force', seed=a.seed, xml=os.path.join(HERE, 'scene_hd.xml'))
    ren = mujoco.Renderer(fw.rob.m, H, W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    os.makedirs(RES, exist_ok=True)
    writer = imageio.get_writer(os.path.join(RES, 'anymal_walk_hd.mp4'), fps=fps, codec='libx264',
                                macro_block_size=8, ffmpeg_log_level='error',
                                output_params=['-crf', '17', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                                               '-movflags', '+faststart', '-profile:v', 'high', '-level', '4.1'])
    state = dict(n=0, gif=[], gif_step=max(1, int(round(fps / 16.0))))

    def snap(rob, t, fw_):
        cam.distance = 2.9; cam.elevation = -14.0
        cam.azimuth = 128.0 + 6.0 * np.sin(0.25 * t)                       # slow drift keeps the scene alive
        cam.lookat[:] = rob.d.qpos[:3] + np.array([0.15, 0, -0.14])
        ren.update_scene(rob.d, cam)
        scn = ren.scene
        pf = rob.foot_pos()
        for li in range(4):                                                 # commanded ground reaction forces
            arrow(scn, pf[li], pf[li] + fw_.f_cmd[li] / 420.0, GREEN, 0.019)
        com = rob.d.qpos[:3].copy()
        F = push(t)
        if np.linalg.norm(F) > 0:                                           # applied disturbance
            arrow(scn, com - F / F_PUSH * 0.62, com, RED, 0.030)
        dh = fw_.w_hat[:3] if fw_.w_hat is not None else np.zeros(3)
        if np.linalg.norm(dh) > 6.0:                                        # estimated disturbance
            arrow(scn, com + np.array([0, 0, 0.12]) - dh / F_PUSH * 0.62, com + np.array([0, 0, 0.12]), ORANGE, 0.024)
        xt = rob.true_state(); xh = fw_.x_hat
        c, sn = np.cos(xh[5]), np.sin(xh[5])
        hud = dict(v=c * xh[6] + sn * xh[7], vc=vcmd(t)[0], roll=xt[3], pitch=xt[4],
                   wz=xh[11], wzc=omcmd(t), z=xt[2], d=dh.copy())
        frame = overlay(ren.render(), t, hud, W, H)      # composite and stream out: nothing is buffered at full size
        writer.append_data(frame)
        n = state['n']; state['n'] = n + 1
        if n % state['gif_step'] == 0:
            state['gif'].append(Image.fromarray(frame).resize((960, 540), Image.LANCZOS)
                                .convert('P', palette=Image.ADAPTIVE, colors=96))
        if abs(t - 0.55 * a.tend) < 0.5 / fps:
            Image.fromarray(frame).save(os.path.join(RES, 'anymal_walk_poster.png'))

    print(f'rendering {W}x{H} at {fps:.1f} fps ...', flush=True)
    fw.run(T_end=a.tend, t_walk=1.0, v_cmd_fn=vcmd, om_cmd_fn=omcmd, push_fn=push, render=dict(every=every, fn=snap))
    writer.close()
    print(f'{state["n"]} frames written', flush=True)
    mp4 = os.path.join(RES, 'anymal_walk_hd.mp4')
    webm = os.path.join(RES, 'anymal_walk_hd.webm')
    os.system(f'ffmpeg -loglevel error -y -i "{mp4}" -c:v libvpx-vp9 -crf 32 -b:v 0 -row-mt 1 '
              f'-pix_fmt yuv420p -an "{webm}"')
    gif = state['gif']
    gif[0].save(os.path.join(RES, 'anymal_walk_preview.gif'), save_all=True, append_images=gif[1:],
                duration=int(1000 * state['gif_step'] / fps), loop=0, optimize=True)
    for p in [mp4, webm, os.path.join(RES, 'anymal_walk_preview.gif'), os.path.join(RES, 'anymal_walk_poster.png')]:
        print(f'{os.path.basename(p):28s} {os.path.getsize(p)/1e6:6.2f} MB')


if __name__ == '__main__':
    main()
