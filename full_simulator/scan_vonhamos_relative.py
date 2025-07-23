"""
Scan three DoFs (pitch, yaw, translation-y) around the *current* aligned
position of the von Hamos crystal, saving PNG frames and a GIF for each DoF.

Usage (values below were used to get bounds for the optimizer):
    python scan_vonhamos_relative.py 10 0.1 0.5 0.5
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------- user‑tunable parameters via CLI -----------------
# nSteps (per side), Δpitch[deg], Δyaw[deg], Δy[mm]
try:
    N, dPitchDeg, dYawDeg, dYmm = map(float, sys.argv[1:5])
    N = int(N)
except ValueError:
    print("Args: <nSteps> <Δpitch_deg> <Δyaw_deg> <Δy_mm>")
    sys.exit(1)

# ------------------------------------------------------------------
import vonhamos_spectrometer as vh
import xrt.runner   as xrtrun
import xrt.backends.raycing.run as rrun
rrun.run_process = vh.run_process

# Folders
root = Path("vonhamos_scan_outputs")
folders = {d: root / d for d in ["pitch", "yaw", "translation"]}
for p in folders.values():
    p.mkdir(parents=True, exist_ok=True)

def run_and_save(beamLine, plots, folder, idx):
    # give each plot its own save path
    for p in plots:
        p.saveName = str(folder / f"frame_{idx:03d}.png")
    xrtrun.run_ray_tracing(plots=plots, backend="raycing",
                           beamLine=beamLine, repeats=1)

def make_gif(folder, gif_name, duration=150):
    pics = sorted(folder.glob("frame_*.png"))
    if not pics:
        return
    frames = [Image.open(p) for p in pics]
    frames[0].save(gif_name, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)

# Build baseline (aligned) beamline once
baseBL = vh.build_beamline()
pitch0 = float(baseBL.vonHamos01.pitch)          # radians
yaw0   = float(getattr(baseBL.vonHamos01, "yaw", 0.0))  # radians (xrt sets yaw lazily)
y0     = float(baseBL.vonHamos01.center[1])      # mm

# Pre‑compute grids
pitch_scan = pitch0 + np.arange(-N, N+1) * np.deg2rad(dPitchDeg)
yaw_scan   = yaw0   + np.arange(-N, N+1) * np.deg2rad(dYawDeg)
y_scan     = y0     + np.arange(-N, N+1) * dYmm

def scan_param(param_name, values, folder):
    plots = vh.define_plots()
    for i, val in enumerate(values):
        bl = vh.build_beamline()
        if param_name == "center_y":
            cx, _, cz = bl.vonHamos01.center
            bl.vonHamos01.center = (cx, val, cz)
        else:
            setattr(bl.vonHamos01, param_name, val)
        run_and_save(bl, plots, folder, i)
        for p in plots:
            p.clean_plots()
    make_gif(folder, folder.parent / f"{folder.name}_scan.gif")

print("pitch scan")
scan_param("pitch", pitch_scan, folders["pitch"])
print("yaw scan")
scan_param("yaw",   yaw_scan,   folders["yaw"])
print("translation‑y scan")
scan_param("center_y", y_scan,  folders["translation"])

print("\nAll scans complete! GIFs are in", root.resolve())
