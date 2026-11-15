"""Pre-registered scoring of frozen models on a set of specimens.

  score.py dev       -> scores every fitted form + baseline + null model on O5-4 (and on the calibration
                        specimens, for the record).  Reads results/fits/*.json and results/baseline.json.
  score.py heldout   -> THE ONE PASS that opens the held-out files.  Requires results/frozen_models.json.
                        Writes results/heldout_access.json (timestamp, hash of the frozen file) first.
Extra modes for the verification gates (same code path):  score.py conv <set>   (mesh + step refinement)
"""
import os, sys, json, time, hashlib, datetime, numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {"dev": [("O5-4", "calibration")], "cal": [("O1-3", "calibration"), ("O14-3", "calibration")],
        "heldout": [("O15-3", "heldout"), ("XT10", "heldout"), ("XR5", "heldout")]}
MESH = {"cal": {"h": 2.5, "h_fine": 1.0}, "fine": {"h": 1.25, "h_fine": 0.5}}
TOL = {"force_nrmse": 0.05, "force_last_err": 0.05, "strain": 0.15, "hot": 0.15}


def _init(threads):
    os.environ["MKL_NUM_THREADS"] = str(threads); os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, HERE)


def _task(spec, role, mesh, models, nsub, tag):
    """models: list of (name, p, elastic_flag).  One Specimen build, several solves."""
    import specimen, data
    S = specimen.Specimen(spec, role, verbose=False, **MESH[mesh])
    out = {}
    for name, p, elastic in models:
        t0 = time.time()
        try:
            sim = S.simulate(np.array(p), nsub=nsub, elastic=elastic)
        except Exception as e:
            out[name] = {"fail": repr(e)}; continue
        m = S.metrics(sim)
        m["time_s"] = time.time() - t0; m["stats"] = sim["stats"]
        m["F"] = sim["F"].tolist()
        os.makedirs(os.path.join(HERE, "results", "cache", "fields"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "results", "cache", "fields", f"{spec}__{name}__{mesh}_n{nsub}{tag}.npz"),
                            E=np.stack([sim["E"][j] for j in data.FIELD_ST]).astype(np.float32), F=sim["F"])
        out[name] = m
    info = {"nn": S.mesh.nn, "ne": S.mesh.ne, "npts": int(S.keep.sum()), "F_meas": S.F_meas.tolist(), "force_ok": S.force_ok.tolist(),
            "ext_st": S.d["ext_st"].tolist(), "xF": float(S.d["xF"]), "Fmax": float(S.d["force"][int(S.d["iF"])]),
            "ylo": S.ylo, "yhi": S.yhi, "T_mean": S.d["T_mean"].tolist(), "bc_rms_max": float(S.d["bc_rms"].max()),
            "drop_stations": np.where(S.d["drop"])[0].tolist()}
    np.savez_compressed(os.path.join(HERE, "results", "cache", "fields", f"{spec}__DIC.npz"), E=S.E_dic.astype(np.float32),
                        pts=S.pts.astype(np.float32), outer=S.outer)
    return spec, out, info


def verdict(m):
    v = {"force_nrmse": abs(m["force_nrmse"]) <= TOL["force_nrmse"], "force_last": abs(m["force_last_err"]) <= TOL["force_last_err"],
         "strain_st13": m["strain_relL2_st13"] <= TOL["strain"], "strain_st27": m["strain_relL2_st27"] <= TOL["strain"],
         "strain_st40": m["strain_relL2_st40"] <= TOL["strain"], "hot": abs(m["hot_err"]) <= TOL["hot"]}
    v["all"] = all(v.values())
    return {k: bool(x) for k, x in v.items()}


def run(specs, models, mesh="cal", nsub=1, tag="", workers=3, threads=4):
    ctx = mp.get_context("spawn")
    res = {}
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(threads,)) as ex:
        futs = [ex.submit(_task, s, r, mesh, models, nsub, tag) for s, r in specs]
        for f in futs:
            spec, out, info = f.result()
            for k, m in out.items():
                if "fail" not in m:
                    m["verdict"] = verdict(m)
            res[spec] = {"models": out, "info": info}
    return res


def fitted_models():
    models = []
    for form in ["M1", "M2", "M3", "M4", "M4b"]:
        f = f"results/fits/{form}.json"
        if os.path.exists(f):
            ok = {k: v for k, v in json.load(open(f))["fits"].items() if "cost" in v}
            if ok:
                models.append((form, ok[min(ok, key=lambda k: ok[k]["cost"])]["p"], False))
    b = json.load(open("results/baseline.json"))["fits"]
    for form in ["M1", "M2", "M3"]:
        models.append((f"uniaxial-{form}", b[form]["p"], False))
    models.append(("null-elastic", [300.0, 0, 1, 0, 0, 1, 1, 0.5, 3], True))
    return models


def show(res):
    for spec, r in res.items():
        print(f"== {spec}  (mesh {r['info']['nn']} nodes, {r['info']['npts']} DIC pts, xF {r['info']['xF']:.2f} mm, Fmax {r['info']['Fmax'] / 1e3:.2f} kN)")
        for k, m in r["models"].items():
            if "fail" in m:
                print(f"  {k:14s} FAILED {m['fail'][:80]}"); continue
            print(f"  {k:14s} F nrmse {100 * m['force_nrmse']:5.2f}%  F last {100 * m['force_last_err']:+6.2f}%  E relL2 "
                  f"{100 * m['strain_relL2_st13']:5.2f}/{100 * m['strain_relL2_st27']:5.2f}/{100 * m['strain_relL2_st40']:5.2f}%  hot {100 * m['hot_err']:+6.2f}% "
                  f"({m['hot_fe']:.3f} vs {m['hot_dic']:.3f})  {'PASS' if m['verdict']['all'] else 'miss'}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        models = fitted_models()
        res = run(SETS["dev"] + SETS["cal"], models, threads=2)
        show(res)
        json.dump(res, open("results/dev_scores.json", "w"), indent=1)
    elif mode == "heldout":
        fz = open("results/frozen_models.json", "rb").read()
        frozen = json.loads(fz)
        stamp = {"opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "frozen_sha256": hashlib.sha256(fz).hexdigest(),
                 "prereg_sha256": hashlib.sha256(open("PREREGISTRATION.md", "rb").read()).hexdigest(), "files": [s for s, _ in SETS["heldout"]]}
        log = json.load(open("results/heldout_access.json")) if os.path.exists("results/heldout_access.json") else []
        log.append(stamp); json.dump(log, open("results/heldout_access.json", "w"), indent=1)
        import data
        for s, r in SETS["heldout"]:
            data.prepare(s, r)
        models = [(k, v["p"], v.get("elastic", False)) for k, v in frozen["models"].items()]
        res = run(SETS["heldout"], models)
        show(res)
        json.dump(res, open("results/heldout_scores.json", "w"), indent=1)
    elif mode == "conv":
        setname = sys.argv[2]
        frozen = json.load(open("results/frozen_models.json"))
        names = sys.argv[3].split(",") if len(sys.argv) > 3 else [frozen["primary"]]
        models = [(k, frozen["models"][k]["p"], False) for k in names]
        out = {}
        for mesh, nsub in [("cal", 2), ("fine", 1)]:
            res = run(SETS[setname], models, mesh=mesh, nsub=nsub, workers=3, threads=4)
            show(res); out[f"{mesh}_n{nsub}"] = res
            json.dump(out, open(f"results/convergence_{setname}.json", "w"), indent=1)
