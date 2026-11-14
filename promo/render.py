"""Promo video, stage 2: compose the MP4 from promo/data/ (stage 1: compute.py).

    python promo/render.py --preview      # 4 fps + contact sheet → promo/out/preview.mp4
    python promo/render.py                # 24 fps + APTUM closing clip → promo/out/vamcore_promo.mp4

Every visual is drawn from the data computed by vamcore (arrays.npz, data.json, the exported frames). Text, layout and
fonts come from the APTUM editorial composer in promo/software-showcase-video/. Only the cross-fades between chapters
and between dose snapshots are editorial transitions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

try:  # in the `vam` conda env NumPy's BLAS only loads after torch has loaded its MKL DLLs (0xc06d007f otherwise)
    import torch  # noqa: F401
except ImportError:
    pass
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

HERE = Path(__file__).resolve().parent
SKILL = HERE / "software-showcase-video"
ASSETS = SKILL / "assets" / "aptum"
CLOSING = ASSETS / "aptum_logo_animation.mp4"
FOOTAGE = HERE / "assets" / "prototype_light_test.mp4"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE / "data"))
ap.add_argument("--out", default=str(HERE / "out"))
ap.add_argument("--preview", action="store_true")
ap.add_argument("--only", type=int, default=None, help="render only this chapter (0 = cover)")
args = ap.parse_args()

DATA = Path(args.data)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
FPS = 4 if args.preview else 24
D = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
A = np.load(DATA / "arrays.npz")

spec = importlib.util.spec_from_file_location("editorial", SKILL / "scripts" / "editorial_frame.py")
editorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editorial)
C = editorial.Composer()
PAL = C.colors
VW, H = C.vw, C.h
PRODUCT = "VAMCORE"

for f in ("LibreFranklin.ttf", "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf"):
    font_manager.fontManager.addfont(str(ASSETS / "fonts" / f))
MONO = font_manager.FontProperties(fname=str(ASSETS / "fonts" / "IBMPlexMono-Regular.ttf"))
BODY = font_manager.FontProperties(fname=str(ASSETS / "fonts" / "LibreFranklin.ttf"))


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


DOSE_CMAP = LinearSegmentedColormap.from_list("dose", [PAL["background"], PAL["mint"], PAL["teal"], PAL["accent"],
                                                       PAL["ink"]])
OCC_CMAP = LinearSegmentedColormap.from_list("occ", [PAL["white"], PAL["mint"], PAL["accent"]])


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fig_to_img(fig) -> Image.Image:
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return img


def new_fig(w=VW, h=H):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(PAL["background"])
    return fig


def fade_visual(img: Image.Image, t_local: float, dur: float, fade=0.4) -> Image.Image:
    a = min(1.0, t_local / fade, (dur - t_local) / fade)
    if a >= 1:
        return img
    bg = Image.new("RGB", img.size, PAL["background"])
    return Image.blend(bg, img, max(0.0, a))


# ── 3D renderer (orientation and voxelization) ────────────────────────────────
pv.OFF_SCREEN = True
MESH_IN = pv.read(str(DATA / "input_tilted.stl"))
MESH_IN = MESH_IN.translate(-np.array(MESH_IN.center), inplace=False)
MESH_OR = pv.read(str(DATA / "job" / "model_oriented.stl"))
MESH_OR = MESH_OR.translate(-np.array(MESH_OR.center), inplace=False)
R_FINAL = np.array(D["orient"]["rotation"])


def rot_about_x_y(frac):
    """Editorial interpolation of the computed rotation R = Ry·Rx: (rx, ry) scaled by frac, applied x then y.
    The rotation itself is done by VTK (numpy matmul crashes once VTK is loaded in this environment)."""
    return D["orient"]["rx"] * frac, D["orient"]["ry"] * frac


def render_mesh(mesh, rot=None, size=(760, 980), plane_z=None, vial_r=None, height=None):
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(PAL["background"])
    m = mesh.copy()
    if rot is not None:
        m = m.rotate_x(rot[0], point=(0, 0, 0), inplace=False).rotate_y(rot[1], point=(0, 0, 0), inplace=False)
    pl.add_mesh(m, color=PAL["accent"], smooth_shading=True, specular=0.3, ambient=0.25)
    hz = height or 25.0
    if vial_r:
        cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=vial_r, height=hz, resolution=96,
                          capping=False)
        pl.add_mesh(cyl, color=PAL["teal"], opacity=0.10)
        for z in (-hz / 2, hz / 2):
            pl.add_mesh(pv.Circle(radius=vial_r, resolution=96).translate((0, 0, z), inplace=False).extract_feature_edges(),
                        color=PAL["teal"], line_width=2)
        pl.add_mesh(pv.Line((0, 0, -hz / 2 - 1.5), (0, 0, hz / 2 + 1.5)), color=PAL["subtle"], line_width=1.5)
    if plane_z is not None:
        side = 2.3 * vial_r if vial_r else 1.25 * max(mesh.bounds[1] - mesh.bounds[0], mesh.bounds[3] - mesh.bounds[2])
        sq = pv.Plane(center=(0, 0, plane_z), direction=(0, 0, 1), i_size=side, j_size=side)
        pl.add_mesh(sq, color=PAL["teal"], opacity=0.35)
        pl.add_mesh(sq.extract_feature_edges(), color=PAL["teal"], line_width=2)
    pl.camera_position = [(48, -38, 14), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(0.82 if vial_r else 1.0)
    img = pl.screenshot(return_img=True)
    pl.close()
    return Image.fromarray(img)


def canvas():
    return Image.new("RGB", (VW, H), PAL["background"])


# ── chapter visuals ───────────────────────────────────────────────────────────
def v_orient(t, dur):
    frac = ease((t - 0.8) / (dur - 2.0))
    img = render_mesh(MESH_IN, rot_about_x_y(frac), vial_r=10.0, size=(760, 820))
    cv = canvas()
    cv.paste(img, ((VW - img.width) // 2, 135))
    d = ImageDraw.Draw(cv)
    o = D["orient"]
    w = o["width_initial_mm"] + (o["width_mm"] - o["width_initial_mm"]) * frac
    face = C.font(22, "mono")
    d.text((76, 980), f"mean projected width  {w:5.2f} mm", font=face, fill=PAL["muted"])
    d.text((76, 1012), f"Rx {o['rx'] * frac:+6.1f}°   Ry {o['ry'] * frac:+6.1f}°", font=face, fill=PAL["subtle"])
    return cv


Z_IDX = A["z_idx"]
TZ = A["target_z"]
GRID = D["job"]["grid"]


def v_voxelize(t, dur):
    frac = ease((t - 0.5) / (dur - 1.4))
    k = int(round((1 - frac) * (len(Z_IDX) - 1)))  # top → bottom
    zmin, zmax = MESH_OR.bounds[4], MESH_OR.bounds[5]
    zp = zmin + (zmax - zmin) * (Z_IDX[k] + 0.5) / GRID[2]
    img = render_mesh(MESH_OR, plane_z=zp, size=(560, 900), vial_r=None)
    cv = canvas()
    cv.paste(img, (20, 120))
    sl = TZ[:, :, k].astype(np.float32).T / 255.0  # [y, x]
    fig = new_fig(560, 640)
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.82])
    ax.imshow(sl, cmap=OCC_CMAP, vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PAL["rule"])
    part = np.count_nonzero((sl > 0.02) & (sl < 0.98))
    fig.text(0.05, 0.04, f"layer {Z_IDX[k] + 1}/{GRID[2]} · {part} partial voxels",
             fontproperties=MONO, fontsize=15, color=PAL["muted"])
    cv.paste(fig_to_img(fig), (600, 230))
    return cv


VIAL = D["vial"]


def v_vial(t, dur):
    Ro, Ri, rp = VIAL["outer_diameter_mm"] / 2, VIAL["r_inner_mm"], VIAL["printable_radius_mm"]
    fig = new_fig()
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    ax.set_aspect("equal")
    ax.set_xlim(-17, 13.5), ax.set_ylim(-13.5, 13.5)
    ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(Ro * np.cos(th), Ro * np.sin(th), color=PAL["white"], zorder=1)
    ax.fill(Ri * np.cos(th), Ri * np.sin(th), color=PAL["background"], zorder=1)
    ax.plot(Ro * np.cos(th), Ro * np.sin(th), color=PAL["ink"], lw=1.6, zorder=3)
    ax.plot(Ri * np.cos(th), Ri * np.sin(th), color=PAL["ink"], lw=1.2, zorder=3)
    zone = ease((t - 3.6) / 1.2)
    if zone > 0:
        ax.fill(rp * np.cos(th), rp * np.sin(th), color=PAL["mint"], alpha=0.55 * zone, zorder=2)
        ax.plot(rp * np.cos(th), rp * np.sin(th), color=PAL["teal"], lw=1.4, ls="--", alpha=zone, zorder=4)
        ax.text(0, -rp + 0.8, f"printable  Ø {2 * rp:.1f} mm", ha="center", fontproperties=MONO, fontsize=15,
                color=PAL["ink"], alpha=zone, zorder=6)
    u = np.array(VIAL["ray_u"])
    tt, dl, valid = np.array(VIAL["ray_t"]), np.array(VIAL["ray_delta"]), np.array(VIAL["ray_valid"])
    sel = np.arange(len(u))[::3]
    reveal = ease((t - 0.4) / 2.6)
    shown = int(round(reveal * len(sel)))
    for i in sel[:shown]:
        y = u[i]
        x0 = -16.5
        if abs(y) >= Ro:
            ax.plot([x0, 13], [y, y], color=PAL["rule"], lw=1, zorder=2)
            continue
        xo = -math.sqrt(Ro * Ro - y * y)
        if not valid[i]:
            ax.plot([x0, xo], [y, y], color=PAL["subtle"], lw=1, alpha=0.6, zorder=5)
            continue
        d = np.array([math.cos(dl[i]), -math.sin(dl[i])])
        n = np.array([math.sin(dl[i]), math.cos(dl[i])])
        c = tt[i] * n
        h = math.sqrt(max(Ri * Ri - tt[i] ** 2, 0))
        pin, pout = c - h * d, c + h * d
        ax.plot([x0, xo], [y, y], color=PAL["accent"], lw=1.3, zorder=5)
        ax.plot([xo, pin[0]], [y, pin[1]], color=PAL["teal"], lw=1.3, zorder=5)
        ax.plot([pin[0], pout[0]], [pin[1], pout[1]], color=PAL["accent"], lw=1.3, alpha=0.85, zorder=5)
    ax.text(-16.5, -12.9, "projector rays (parallel)", fontproperties=MONO, fontsize=14, color=PAL["muted"])
    ax.text(Ro * 0.72, Ro * 0.78, "glass", fontproperties=MONO, fontsize=14, color=PAL["muted"])
    ax.text(-Ri * 0.45, Ri * 0.6, "resin", fontproperties=MONO, fontsize=14, color=PAL["muted"])
    return fig_to_img(fig)


SNAPS = D["snapshots"]
SNAP_IMG = [np.load(DATA / f"snap_{s['iterations']}_xz.npy").astype(np.float32) for s in SNAPS]
FINAL_XZ = A["recon_xz"].astype(np.float32)
TARGET_XZ = A["target_xz"].astype(np.float32)
VER = A["ver_curve"]
N_IT = int(D["job"]["metrics"]["iterations_run"] or len(VER))
STATES = [(s["iterations"], img) for s, img in zip(SNAPS, SNAP_IMG)] + [(N_IT, FINAL_XZ)]


def v_prisma(t, dur):
    step = (dur - 0.6) / len(STATES)
    pos = max(0.0, t - 0.3) / step
    i = min(int(pos), len(STATES) - 1)
    blend = ease((pos - i - 0.75) / 0.25) if i < len(STATES) - 1 else 0.0
    k, img = STATES[i]
    if blend > 0:
        img = (1 - blend) * img + blend * STATES[i + 1][1]
        k = STATES[i + 1][0] if blend > 0.5 else k
    fig = new_fig()
    ax = fig.add_axes([0.06, 0.32, 0.88, 0.55])
    ax.imshow(img.T, cmap=DOSE_CMAP, vmin=0, vmax=1, origin="lower", interpolation="bilinear", aspect="equal")
    ax.contour(TARGET_XZ.T, levels=[0.5], colors=[PAL["teal"]], linewidths=1.0)
    ax.axis("off")
    fig.text(0.5, 0.275, f"vertical section · after {k} iterations", ha="center", fontproperties=MONO,
             fontsize=16, color=PAL["ink"])
    ax2 = fig.add_axes([0.10, 0.07, 0.82, 0.17])
    it = np.arange(len(VER))
    ok = np.isfinite(VER)
    ax2.plot(it[ok], VER[ok], color=PAL["accent"], lw=1.8)
    cur = min(k, len(VER) - 1)
    near = it[ok][np.argmin(np.abs(it[ok] - cur))]
    ax2.plot([near], [VER[near]], "o", color=PAL["teal"], ms=8)
    ax2.set_yscale("log")
    ax2.set_xlim(0, max(N_IT, 10))
    ax2.set_facecolor(PAL["background"])
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(PAL["rule"])
    ax2.tick_params(colors=PAL["subtle"], labelsize=11)
    ax2.set_xlabel("iteration", fontproperties=MONO, fontsize=13, color=PAL["subtle"])
    ax2.set_ylabel("VER", fontproperties=MONO, fontsize=13, color=PAL["subtle"])
    return fig_to_img(fig)


HG, HV, HB = A["hist_gel"].astype(float), A["hist_void"].astype(float), A["hist_bins"]
ROB = D["robustness"]


def v_verify(t, dur):
    grow = ease(t / 1.6)
    fig = new_fig()
    ax = fig.add_axes([0.09, 0.42, 0.84, 0.42])
    ctr = (HB[:-1] + HB[1:]) / 2
    w = HB[1] - HB[0]
    gmin, vmax = D["job"]["gel_min"], D["job"]["void_max"]
    if gmin > vmax:
        ax.axvspan(vmax, gmin, color=PAL["mint"], alpha=0.6, lw=0)
        ax.text((vmax + gmin) / 2, 1.02, "process window", transform=ax.get_xaxis_transform(), ha="center",
                fontproperties=MONO, fontsize=13, color=PAL["ink"])
    ax.bar(ctr, np.maximum(HV * grow, 0.8), width=w, color=PAL["teal"], alpha=0.85, label="void (must stay liquid)")
    ax.bar(ctr, np.maximum(HG * grow, 0.8), width=w, color=PAL["accent"], alpha=0.9, label="gel (must cure)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(HG.max(), HV.max()) * 3)
    ax.set_xlim(0, 1)
    ax.set_facecolor(PAL["background"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["rule"])
    ax.tick_params(colors=PAL["subtle"], labelsize=11)
    ax.set_xlabel("normalized dose", fontproperties=MONO, fontsize=13, color=PAL["subtle"])
    ax.set_ylabel("voxels", fontproperties=MONO, fontsize=13, color=PAL["subtle"])
    ax.legend(prop=BODY, frameon=False, loc="upper center", fontsize=13)
    sc = ROB["scenarios"]
    shown = int(round(ease((t - 1.8) / (dur - 2.6)) * len(sc)))
    fig.text(0.09, 0.30, "robustness test · 20 perturbed scenarios", fontproperties=MONO, fontsize=15,
             color=PAL["muted"])
    for j, s in enumerate(sc):
        x0 = 0.09 + (j % 10) * 0.084
        y0 = 0.19 - (j // 10) * 0.085
        on = j < shown
        col = (PAL["accent"] if s["pass"] else "#C0392B") if on else PAL["rule"]
        fig.patches.append(matplotlib.patches.FancyBboxPatch((x0, y0), 0.07, 0.06, boxstyle="round,pad=0.004",
                                                             transform=fig.transFigure, color=col, alpha=0.9 if on else 0.6))
        if on:
            fig.text(x0 + 0.035, y0 + 0.03, f"{s['shape_error_um']:.0f} µm", ha="center", va="center",
                     fontproperties=MONO, fontsize=11, color=PAL["white"])
    return fig_to_img(fig)


FRAMES = sorted((DATA / "job").glob("frame_*.png"))


def v_stack(t, dur):
    frac = min(1.0, max(0.0, (t - 0.3) / (dur - 0.9)))
    i = min(int(frac * len(FRAMES)), len(FRAMES) - 1)
    fr = Image.open(FRAMES[i]).convert("L")
    fr.thumbnail((760, 760), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (fr.width + 40, fr.height + 40), PAL["ink"])
    card.paste(Image.merge("RGB", (fr, fr, fr)), (20, 20))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width, card.height), radius=18, fill=255)
    cv = canvas()
    x0, y0 = (VW - card.width) // 2, 150
    cv.paste(card, (x0, y0), mask)
    d = ImageDraw.Draw(cv)
    d.text((x0, y0 + card.height + 18), f"frame {i + 1:03d}/{len(FRAMES)}  ·  angle {i * 360 / len(FRAMES):5.1f}°",
           font=C.font(22, "mono"), fill=PAL["muted"])
    return cv


FOOT = imageio.mimread(str(FOOTAGE), memtest=False) if FOOTAGE.exists() else []
FOOT_FPS = 30.0
try:
    FOOT_FPS = imageio.get_reader(str(FOOTAGE)).get_meta_data().get("fps", 30.0)
except Exception:
    pass


def v_printer(t, dur):
    cv = canvas()
    if not FOOT:
        return cv
    fr = Image.fromarray(FOOT[min(int(t * FOOT_FPS), len(FOOT) - 1)])
    fr.thumbnail((1050, 900), Image.Resampling.LANCZOS)
    mask = Image.new("L", fr.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, fr.width, fr.height), radius=18, fill=255)
    cv.paste(fr, ((VW - fr.width) // 2, (H - fr.height) // 2), mask)
    return cv


# ── chapters ──────────────────────────────────────────────────────────────────
J, O, M = D["job"], D["orient"], D["job"]["metrics"]
st = J["stage_times"]
CH = [
    dict(label="Orient", dur=6.5, visual=v_orient, title="Finds the\nnarrowest pose",
         body=["The tilted input is turned until it is", "as narrow as possible around the axis,",
               "within the vial's printable zone."],
         metric=(f"−{O['reduction_pct']:.0f} %", f"mean projected width, {O['width_initial_mm']:.1f} to {O['width_mm']:.1f} mm"),
         evidence=f"input tilted {D['provenance']['params']['tilt_deg']:.0f}° for the demo · computed orientation"),
    dict(label="Voxelize", dur=7.0, visual=v_voxelize, title="Layer by layer,\npartial occupancy",
         body=["Each layer is sliced once and filled", "along scanlines. Edge voxels keep",
               "their exact fraction of material."],
         metric=(f"{st.get('voxelize', 0):.1f} s", f"{J['grid'][2]} layers · previous version {D['historical']['voxelize_512_previous_s']:.0f} s (historical)"),
         evidence="real cross-sections of the job"),
    dict(label="Vial", dur=6.5, visual=v_vial, title="No index-matching\nbath needed",
         body=["Refraction and losses at both vial walls", "are traced with Snell and Fresnel,",
               "inside the optimization."],
         metric=(f"Ø {VIAL['printable_diameter_mm']:.1f} mm", f"printable zone in a Ø{VIAL['outer_diameter_mm']:.0f} mm vial, {VIAL['wall_thickness_mm']:.0f} mm wall"),
         evidence="computed optics · physical validation pending"),
    dict(label="Prisma", dur=7.5, visual=v_prisma, title="Optimizes what the\nprojector emits",
         body=["Prisma, our GPU projector, puts the", "vial and the resin inside the model,",
               "so every step refines the real image."],
         metric=(f"{J['elapsed_s']:.1f} s", f"optimization · {J['grid'][2]} layers · 360 angles · {N_IT} iterations"),
         evidence="simulated dose · Prisma engine"),
    dict(label="Verify", dur=6.5, visual=v_verify, title="Checks the print\nbefore you print",
         body=[f"Exposure tolerance ±{M.get('exposure_tolerance_pct') or 0:.1f} %. Twenty",
               "perturbed runs: axis, speed, blur,", "LED flicker and dose."],
         metric=(f"{J['score_bio']:.0f}/100", "Score Bio · simulated dose quality, not a print test"),
         evidence="simulated · not a print test"),
    dict(label="Stack", dur=6.0, visual=v_stack, title="360 frames, ready\nfor the projector",
         body=["One image per degree at the projector's", "physical scale, cropped to the useful",
               "area and ready to stream."],
         metric=(f"{J['total_s']:.0f} s", "from STL to frames, robustness test included"),
         evidence="exported frames of the job"),
    dict(label="Printer", dur=5.6, visual=v_printer, title="Plays back in sync\nwith the rotation",
         body=["The stack streams to the projector and", "the stage turns one degree per frame.",
               "Shown: an early light test."],
         metric=None, evidence="prototype light test · real footage · no part formed"),
]
COVER_S = 3.0
GPU = D["machine"]["gpu"] or "CPU"
for c in CH:
    if c["label"] in ("Prisma", "Stack"):
        c["metric"] = (c["metric"][0], c["metric"][1] + f" · {GPU.replace('NVIDIA GeForce ', '')}")


def cover():
    img = C.opening(PRODUCT, "From STL to projection, physically correct",
                    kicker="TOMOGRAPHIC VOLUMETRIC PRINTING", footer="Beta · simulated results")
    d = ImageDraw.Draw(img)
    C.text(d, (C.w / 2, 745), "by APTUM", C.font(26, "mono"), PAL["accent"], anchor="mm")
    return img


def chapter_frame(i, t):
    c = CH[i]
    vis = fade_visual(c["visual"](t, c["dur"]), t, c["dur"])
    metric = c["metric"]
    try:
        return C.chapter(visual=vis, index=i + 1, count=len(CH), label=c["label"], title=c["title"], body=c["body"],
                         product=PRODUCT, metric=metric, evidence_label=c["evidence"], beta=True, t=t / c["dur"])
    except ValueError:
        # caption too long for the column: drop the trailing qualifier rather than shrinking the system
        metric = (metric[0], metric[1].rsplit(" · ", 1)[0]) if metric else None
        return C.chapter(visual=vis, index=i + 1, count=len(CH), label=c["label"], title=c["title"], body=c["body"],
                         product=PRODUCT, metric=metric, evidence_label=c["evidence"], beta=True, t=t / c["dur"])


def frames():
    if args.only in (None, 0):
        img = cover()
        img.save(OUT / "cover.png")
        for _ in range(int(COVER_S * FPS)):
            yield "cover", img
    for i, c in enumerate(CH):
        if args.only not in (None, i + 1):
            continue
        n = int(round(c["dur"] * FPS))
        for f in range(n):
            yield c["label"], chapter_frame(i, (f + 0.5) / FPS)


def main():
    name = "preview" if args.preview else "vamcore_promo"
    body_mp4 = OUT / f"{name}_body.mp4"
    w = imageio.get_writer(str(body_mp4), fps=FPS, codec="libx264", quality=None,
                           ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium"], macro_block_size=8)
    contacts, chapters, t0, last = [], [], 0.0, None
    nf = 0
    for label, img in frames():
        if label != last:
            if last is not None:
                chapters.append({"label": last, "start": t0, "end": nf / FPS})
            t0, last = nf / FPS, label
            sampled = 0
        w.append_data(np.asarray(img))
        if args.preview and nf % 3 == 0:
            contacts.append(img.resize((480, 270), Image.Resampling.LANCZOS))
        nf += 1
        if nf % (FPS * 5) == 0:
            print(f"  {nf / FPS:5.1f} s rendered", flush=True)
    chapters.append({"label": last, "start": t0, "end": nf / FPS})
    w.close()
    out = OUT / f"{name}.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if args.only is None and CLOSING.exists():
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(body_mp4), "-i", str(CLOSING), "-filter_complex",
                        f"[1:v]fps={FPS},scale=1920:1080,setsar=1,format=yuv420p[c];[0:v][c]concat=n=2:v=1:a=0[v]",
                        "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out)], check=True)
        chapters.append({"label": "APTUM closing", "start": nf / FPS, "end": nf / FPS + 6.0})
        body_mp4.unlink()
    else:
        body_mp4.replace(out)
    (OUT / f"{name}_chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    if contacts:
        cols = 6
        rows = math.ceil(len(contacts) / cols)
        sheet = Image.new("RGB", (cols * 480, rows * 270), PAL["white"])
        for j, c in enumerate(contacts):
            sheet.paste(c, ((j % cols) * 480, (j // cols) * 270))
        sheet.save(OUT / f"{name}_contacts.png")
    print(f"[done] {out} · {nf / FPS:.1f} s + closing")


if __name__ == "__main__":
    main()
