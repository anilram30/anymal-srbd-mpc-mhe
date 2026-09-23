# Media

- `anymal_walk_hd.mp4`  — closed-loop run, 1920×1080 @ 30 fps, H.264 (web-ready, faststart)
- `anymal_walk_hd.webm` — same, VP9
- `anymal_walk_preview.gif` — 720-wide looping preview for README/embeds
- `anymal_walk_poster.png`  — poster frame (the push instant)
- `contact_sheet.png`       — eight frames across the run

Regenerate with `MUJOCO_GL=osmesa python3 sim/render_hd.py --fps 30`, or render in
bounded-memory 2 s segments with `sim/render_seg.py --t0 0 --t1 2 --idx 0` (etc.) and
join them with `ffmpeg -f concat`
(needs the solver libraries; see interface/README.md).
