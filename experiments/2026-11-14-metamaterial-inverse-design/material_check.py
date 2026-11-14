"""Check that CalculiX *HYPERELASTIC, NEO HOOKE equals fem.psi: one CPE8 element, F = diag(lam, 1) and simple shear."""
import subprocess, json, re, numpy as np, jax, jax.numpy as jnp
import fem, ccx
from pathlib import Path
wd = Path("results/ccx/material_check")
P = jax.grad(fem.psi)
res = []
for mode, val in [("stretch", 1.4), ("stretch", 0.7), ("shear", 0.5)]:
    nodes = [(0,0),(1,0),(1,1),(0,1),(.5,0),(1,.5),(.5,1),(0,.5)]
    F = np.array([[val, 0], [0, 1.0]]) if mode == "stretch" else np.array([[1.0, val], [0, 1.0]])
    out = ["*NODE, NSET=NALL"] + [f"{i+1}, {x}, {y}, 0." for i, (x, y) in enumerate(nodes)]
    out += ["*ELEMENT, TYPE=CPE8, ELSET=EALL", "1, 1,2,3,4,5,6,7,8", "*NSET, NSET=RIGHT", "2,3,6", "*NSET,NSET=TOP", "3,4,7",
            "*MATERIAL, NAME=NH", "*HYPERELASTIC, NEO HOOKE", f"{fem.C10}, {fem.D1:.10g}", "*SOLID SECTION, ELSET=EALL, MATERIAL=NH", "1.0",
            "*STEP, NLGEOM, INC=1000", "*STATIC", "0.05, 1., 1e-8, 0.05", "*BOUNDARY"]
    for i, (x, y) in enumerate(nodes):
        u = (F - np.eye(2)) @ np.array([x, y])
        out += [f"{i+1}, 1, 1, {u[0]:.10g}", f"{i+1}, 2, 2, {u[1]:.10g}"]
    out += ["*NODE PRINT, NSET=RIGHT, TOTALS=ONLY", "RF", "*NODE PRINT, NSET=TOP, TOTALS=ONLY", "RF", "*END STEP"]
    (wd / "job.inp").write_text("\n".join(out) + "\n")
    subprocess.run([ccx.MM, "run", "-n", "solidmech", "ccx", "job"], cwd=wd, capture_output=True, env=ccx.ENV)
    txt = (wd / "job.dat").read_text()
    blocks = re.findall(r"total force.*?set (\w+) and time\s+([0-9.Ee+-]+)\s*\n\s*\n?\s*([-0-9.Ee+]+)\s+([-0-9.Ee+]+)", txt)
    last = {b[0]: (float(b[2]), float(b[3])) for b in blocks}
    Pk = np.asarray(P(jnp.asarray(F)))
    # traction on right face (normal e_x): P[:,0]; on top face (normal e_y): P[:,1]
    r = dict(mode=mode, value=val, ccx_right=last["RIGHT"], jax_right=Pk[:, 0].tolist(), ccx_top=last["TOP"], jax_top=Pk[:, 1].tolist())
    r["max_rel_diff"] = float(max(np.max(np.abs(np.array(last["RIGHT"]) - Pk[:, 0])), np.max(np.abs(np.array(last["TOP"]) - Pk[:, 1]))) / np.max(np.abs(Pk)))
    res.append(r); print(r)
json.dump(res, open("results/material_check.json", "w"), indent=1)
for ext in ("frd", "12d", "cvg", "sta"):
    f = wd / f"job.{ext}"
    if f.exists(): f.unlink()
