"""Slab (one element through the width, ~plane stress) calibration sweep: l x geometry, Gc fixed nominal.
Loads scale as sqrt(Gc) at fixed l, so ratios between geometries are what identifies l."""
import json, sys, numpy as np, dmc
out = {}
for l, h in ((0.5, 0.25), (1.0, 0.5), (2.0, 0.5), (4.0, 0.5)):
    for geom in ("HC", "HB", "HA"):
        Gc = 0.06
        P_guess = {"HC": 230, "HB": 270, "HA": 420}[geom]
        r = dmc.run(geom, h, 1, l, Gc, ddelta=P_guess / 1780.0 / 70, tag=f"cal2d_{geom}_l{l}", verbose=False)
        out[f"{geom}_l{l}"] = r
        json.dump(out, open("results/dmc/calib2d_sweep.json", "w"), indent=1)
