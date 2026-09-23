# Copyright 2026 Sreeram Anil
# SPDX-License-Identifier: Apache-2.0
#
# Part of the unified SRBD NMPC + moving-horizon estimation framework for ANYmal-C
# by Sreeram Anil. Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# If you use this file, this attribution to Sreeram Anil must be retained.

"""Render one time-segment of the walk to its own mp4 (bounded memory)."""
import os, sys, argparse, numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image
import render_hd as R   # reuse overlay(), arrow(), phase_of(), constants
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,'..','results','segments'); os.makedirs(RES,exist_ok=True)
from framework import Framework

ap=argparse.ArgumentParser(); ap.add_argument('--t0',type=float); ap.add_argument('--t1',type=float)
ap.add_argument('--fps',type=int,default=30); ap.add_argument('--idx',type=int); a=ap.parse_args()
W,H=1920,1080; every=max(1,int(round(1.0/(a.fps*0.002)))); fps=1.0/(every*0.002)
fw=Framework(noise=True,dist_comp='force',seed=0,xml=os.path.join(HERE,'scene_hd.xml'))
ren=mujoco.Renderer(fw.rob.m,H,W)
ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=1; ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
out=os.path.join(RES,f'seg{a.idx:02d}.mp4')
wr=imageio.get_writer(out,fps=fps,codec='libx264',macro_block_size=8,ffmpeg_log_level='error',
    output_params=['-crf','17','-preset','medium','-pix_fmt','yuv420p','-profile:v','high','-level','4.1'])
st={'n':0}
def snap(rob,t,fw_):
    if t < a.t0 - 1e-9 or t >= a.t1 - 1e-9: return
    cam.distance=2.9; cam.elevation=-14.0; cam.azimuth=128.0+6.0*np.sin(0.25*t)
    cam.lookat[:]=rob.d.qpos[:3]+np.array([0.15,0,-0.14]); ren.update_scene(rob.d,cam); scn=ren.scene
    pf=rob.foot_pos()
    for li in range(4): R.arrow(scn,pf[li],pf[li]+fw_.f_cmd[li]/420.0,R.GREEN,0.019)
    com=rob.d.qpos[:3].copy(); F=R.push(t)
    if np.linalg.norm(F)>0: R.arrow(scn,com-F/R.F_PUSH*0.62,com,R.RED,0.030)
    dh=fw_.w_hat[:3] if fw_.w_hat is not None else np.zeros(3)
    if np.linalg.norm(dh)>6.0: R.arrow(scn,com+[0,0,0.12]-dh/R.F_PUSH*0.62,com+[0,0,0.12],R.ORANGE,0.024)
    xt=rob.true_state(); xh=fw_.x_hat; c,sn=np.cos(xh[5]),np.sin(xh[5])
    hud=dict(v=c*xh[6]+sn*xh[7],vc=R.vcmd(t)[0],roll=xt[3],pitch=xt[4],wz=xh[11],wzc=R.omcmd(t),z=xt[2],d=dh.copy())
    fr=R.overlay(ren.render(),t,hud,W,H); wr.append_data(fr); st['n']+=1
    if abs(t-4.0)<0.5/fps: Image.fromarray(fr).save(os.path.join(HERE,'..','results','anymal_walk_poster.png'))
fw.run(T_end=a.t1,t_walk=1.0,v_cmd_fn=R.vcmd,om_cmd_fn=R.omcmd,push_fn=R.push,render=dict(every=every,fn=snap))
wr.close(); print(f'seg{a.idx} [{a.t0},{a.t1}) {st["n"]} frames -> {out} {os.path.getsize(out)/1e6:.2f} MB',flush=True)
