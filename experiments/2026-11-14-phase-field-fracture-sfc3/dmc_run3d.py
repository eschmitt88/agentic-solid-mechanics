"""3D runs (full width resolved) for the four calibration geometries and the challenge geometry."""
import json, sys, numpy as np, dmc
l = float(sys.argv[1]); h = float(sys.argv[2]); ny = int(sys.argv[3]); Gc = float(sys.argv[4]); geoms = sys.argv[5].split(",")
suffix = sys.argv[6] if len(sys.argv) > 6 else ""
guess = {"HC": 215, "HB": 255, "HA": 410, "H45": 280, "CH": 320}
import os
stop = float(os.environ.get("DMC_STOP", "0.25")); nsteps = float(os.environ.get("DMC_NSTEPS", "70"))
for geom in geoms:
    dmc.run(geom, h, ny, l, Gc, ddelta=guess[geom] / 1780.0 / nsteps, tag=f"r3d_{geom}_l{l}_h{h}_ny{ny}{suffix}", verbose=True,
            stop_frac=stop, max_steps=int(6 * nsteps))
