"""Independent re-run of an identified von Mises model in CalculiX 2.23 (CPS6, NLGEOM, tabulated isotropic
hardening), same mesh and the same DIC-driven boundary displacements as the JAX solver.
usage: ccx_check.py SPEC ROLE MODEL_NAME   (model read from results/frozen_models.json)"""
import os, sys, json, subprocess, time, re, numpy as np
import specimen, data, femu, fem

CCX = ["env", "MAMBA_ROOT_PREFIX=" + os.path.expanduser("~/projects/.micromamba"), "OMP_NUM_THREADS=4",
       os.path.expanduser("~/.local/bin/micromamba"), "run", "-n", "solidmech", "ccx"]


def write_deck(S, p, path, nst=data.NST):
    nd, el = S.mesh.nodes, S.mesh.elems
    L = ["*NODE, NSET=NALL"]
    L += [f"{i + 1}, {x:.10g}, {y:.10g}, 0." for i, (x, y) in enumerate(nd)]
    L.append("*ELEMENT, TYPE=CPS6, ELSET=EALL")
    L += [f"{e + 1}, " + ", ".join(str(n + 1) for n in c) for e, c in enumerate(el)]
    L.append("*NSET, NSET=NTOP"); L += [str(n + 1) for n in S.top]
    L.append("*NSET, NSET=NBOT"); L += [str(n + 1) for n in S.bot]
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{fem.ELASTIC['E']}, {fem.ELASTIC['nu']}", "*PLASTIC"]
    a = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 400)])
    L += [f"{s:.8g}, {x:.8g}" for x, s in zip(a, femu.sigy(p, a))]
    L += ["*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL", f"{specimen.THICK}"]
    bn = np.concatenate([S.bot, S.top]); nb, nt = len(S.bot), len(S.top)
    for j in range(1, nst):
        ub = S.ubar_st[j]
        ux = np.concatenate([ub[:nb], ub[2 * nb: 2 * nb + nt]]); uy = np.concatenate([ub[nb: 2 * nb], ub[2 * nb + nt:]])
        L += ["*STEP, NLGEOM, INC=1000", "*STATIC", "0.25, 1.0, 1e-6, 0.5", "*BOUNDARY"]
        for n, a_, b_ in zip(bn, ux, uy):
            L.append(f"{n + 1}, 1, 1, {a_:.10g}"); L.append(f"{n + 1}, 2, 2, {b_:.10g}")
        L += ["*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=10000", "RF"]
        if j in data.FIELD_ST:
            L += ["*NODE PRINT, NSET=NALL, FREQUENCY=10000", "U"]
        L.append("*END STEP")
    open(path, "w").write("\n".join(L) + "\n")


def parse_dat(path, nn):
    txt = open(path).read()
    blocks = re.split(r"\n\s*\n(?= [a-z])", txt)
    F = []; U = []
    for b in re.split(r"\n(?= total force| displacements)", txt):
        if b.lstrip().startswith("total force"):
            lines = [l for l in b.splitlines()[1:] if l.strip()]
            F.append(float(lines[0].split()[1]))
        elif b.lstrip().startswith("displacements"):
            lines = [l.split() for l in b.splitlines()[1:] if l.strip()]
            arr = np.array([[float(x) for x in l[:4]] for l in lines if len(l) >= 4 and l[0].isdigit()][:nn])
            u = np.zeros((nn, 2)); u[arr[:, 0].astype(int) - 1] = arr[:, 1:3]
            U.append(u)
    return np.array(F), U


if __name__ == "__main__":
    spec, role, name = sys.argv[1:4]
    frozen = json.load(open(sys.argv[4] if len(sys.argv) > 4 else "results/frozen_models.json"))
    p = np.array(frozen["models"][name]["p"])
    assert abs(p[6] - 1) < 1e-12 and abs(p[7] - 0.5) < 1e-12 and abs(p[8] - 3) < 1e-12, "CalculiX check needs a von Mises model"
    S = specimen.Specimen(spec, role)
    wd = os.path.join("results", "ccx"); os.makedirs(wd, exist_ok=True)
    job = f"{spec}_{name}"
    write_deck(S, p, os.path.join(wd, job + ".inp"))
    t0 = time.time()
    r = subprocess.run(CCX + [job], cwd=wd, capture_output=True, text=True)
    open(os.path.join(wd, job + ".log"), "w").write(r.stdout + r.stderr)
    print("ccx wall %.0fs" % (time.time() - t0), r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    F, U = parse_dat(os.path.join(wd, job + ".dat"), S.mesh.nn)
    sim = S.simulate(p)
    Fj = sim["F"][1:]
    n = min(len(F), len(Fj))
    ok = np.arange(1, n + 1) >= data.FORCE_ST0
    out = {"spec": spec, "model": name, "n_steps_ccx": int(len(F)), "ccx_wall_s": time.time() - t0,
           "force_nrmse_jax_vs_ccx": float(np.sqrt(np.mean((F[:n][ok] - Fj[:n][ok]) ** 2)) / np.mean(np.abs(Fj[:n][ok]))),
           "force_max_rel_diff": float(np.max(np.abs(F[:n][ok] / Fj[:n][ok] - 1))), "F_ccx": F.tolist(), "F_jax": Fj.tolist(), "strain_relL2": {}}
    w = np.array([1, 1, 2.0])[:, None]
    Ec = {}
    for i, j in enumerate(data.FIELD_ST):
        if i < len(U):
            E, _ = specimen.gl_strain(S.Gx, S.Gy, S.A @ U[i][:, 0], S.A @ U[i][:, 1]); Ec[j] = E
            Ej = sim["E"][j]
            out["strain_relL2"][str(j)] = float(np.sqrt((w * (E - Ej) ** 2).sum() / (w * Ej ** 2).sum()))
    # CalculiX's own score against the experiment (same metrics)
    simc = {"F": np.concatenate([[0.0], F, np.full(data.NST - 1 - len(F), np.nan)]), "E": Ec}
    if len(Ec) == 3 and len(F) == data.NST - 1:
        out["ccx_metrics_vs_experiment"] = S.metrics(simc)
    out["jax_metrics_vs_experiment"] = S.metrics(sim)
    out["pass"] = bool(out["force_nrmse_jax_vs_ccx"] <= 0.02 and out["strain_relL2"].get("40", 1) <= 0.03)
    json.dump(out, open(f"results/ccx_check_{spec}_{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("F_")}, indent=1))
