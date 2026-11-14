# VAMCORE promo video

A 54.5 s commercial (1920×1080, 24 fps, H.264, no audio) built from one real VAMCORE job, in the APTUM editorial
style of `software-showcase-video/`. It ends with the full APTUM closing clip.

## Reproduce

```bash
python promo/compute.py            # stage 1, GPU: real job + snapshots → promo/data/ (≈ 4 min on an RTX 4060 Ti)
python promo/render.py --preview   # 4 fps preview + contact sheet → promo/out/
python promo/render.py             # final → promo/out/vamcore_promo.mp4 (+ cover.png, chapters JSON)
python promo/software-showcase-video/scripts/verify_video.py promo/out/vamcore_promo.mp4 --out promo/out/qa_final --chapters promo/out/vamcore_promo_chapters.json
```

`promo/data/` and `promo/out/` are not versioned. In the `vam` conda environment, NumPy's BLAS only loads after torch,
so `render.py` imports torch first.

## The job

`vamcore.pipeline.run` on the bundled trifurcated vasculature, tilted 35° about X as the demo input, oriented with
`vamcore.orient.find_orientation` (Rx −35.0°, Ry 0.0°), at 512 layers (281 × 281 × 516 voxels) with 360 angles, a
Ø20 mm borosilicate vial with a 1 mm wall, GelMA resin, vial inside the model and the robustness test on.
Machine: NVIDIA RTX 4060 Ti, PyTorch 2.10, vamcore 0.2.0. Input SHA-256 `6d4a14b98fc3826a…`, vamcore code SHA-256
`7d823cf29a47a409…` (full hashes in `promo/data/data.json`).

## Chapters and figures

| Time | Chapter | What the visual shows | Figure on screen | Source |
|---|---|---|---|---|
| 0–3 s | Cover | Name, claim, "by APTUM" | — | — |
| 3–9.5 s | 01 Orient | Tilted input rotated to the computed pose (the rotation in between is an editorial transition) | −18 % mean projected width (11.3 to 9.3 mm) | `find_orientation` |
| 9.5–16.5 s | 02 Voxelize | A plane sweeps the part; the real occupancy of that layer | 1.4 s · previous version 196 s | this job; 195.5 s is a historical measurement of the previous voxelizer on the same PC (13-11-2026) |
| 16.5–23 s | 03 Vial | Rays traced by `VialOptics.ray_map`; printable zone | Ø 14.8 mm | `VialOptics.printable_radius` |
| 23–30.5 s | 04 Prisma | Vertical dose section after 1, 6, 16, 41, 81 and 161 iterations, synced with the VER curve | 49.5 s optimization, 161 iterations | the job; snapshots are separate runs of the same job with `iterations=k`, cross-fades are editorial |
| 30.5–37 s | 05 Verify | Gel/void dose histogram with the process window; the 20 robustness scenarios (15 pass, 5 fail) with their shape error | Score Bio 86/100 (85.6), tolerance ±2.7 % | the job |
| 37–43 s | 06 Stack | The 360 exported frames (991 × 1347 px) | 74 s STL to frames, robustness included | the job |
| 43–48.5 s | 07 Printer | Real footage of an early prototype light test | — | `assets/prototype_light_test.mp4`; no part was formed |
| 48.5–54.5 s | Closing | APTUM logo animation, unchanged | — | `software-showcase-video/assets/aptum/` |

## Limitations

- Every engine result is simulated; the video shows no printed part and says so.
- The robustness magnitudes are assumptions, not measurements of the printer.
- No audio or voice-over.
