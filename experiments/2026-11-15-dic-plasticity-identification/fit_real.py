"""FEMU fits of the model forms to the two calibration specimens (O1-3, O14-3).
usage: fit_real.py FORM [threads]   -> results/fits/FORM.json"""
import os, sys, json, time, numpy as np
import femu

CAL = [("O1-3", {"h": 2.5, "h_fine": 1.0}), ("O14-3", {"h": 2.5, "h_fine": 1.0})]

def _best(form):
    f = {k: v for k, v in json.load(open(f"results/fits/{form}.json"))["fits"].items() if "cost" in v}
    return f[min(f, key=lambda k: f[k]["cost"])]["p"]


def starts_for(form):
    base = json.load(open("results/baseline.json"))["fits"]
    if form == "M1":
        return {"uniaxial": base["M1"]["p"][:3], "far": [200.0, 600.0, 6.0]}
    if form == "M2":
        return {"uniaxial": [base["M2"]["p"][i] for i in (0, 1, 2, 3)], "far": [220.0, 700.0, 2.0, 400.0], "far2": [450.0, 200.0, 12.0, 1500.0]}
    if form == "M3":
        return {"uniaxial": [base["M3"]["p"][i] for i in (0, 1, 2, 3, 4, 5)], "far": [250.0, 300.0, 30.0, 600.0, 400.0, 2.0],
                "m2like": [330.0, 370.0, 5.3, 880.0, 50.0, 60.0]}
    if form == "M4":
        m2 = _best("M2")
        return {"m2+iso": [m2[0], m2[1], m2[2], m2[3], 1.0, 0.5, 3.0], "far": [250.0, 600.0, 3.0, 600.0, 1.2, 0.4, 2.6],
                "far2": [m2[0], m2[1], m2[2], m2[3], 0.85, 0.6, 3.4]}
    if form == "M4b":
        m3 = _best("M3")
        m4 = _best("M4")
        return {"m3+hill(M4)": m3[:6] + m4[6:9], "m3+iso": m3[:6] + [1.0, 0.5, 3.0]}

if __name__ == "__main__":
    form = sys.argv[1]; threads = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs("results/fits", exist_ok=True)
    pool = femu.Pool(CAL, threads=threads)
    out = {"form": form, "fits": {}}
    for name, p0 in starts_for(form).items():
        print(f"[{form}] start {name}: {np.round(p0, 3).tolist()}", flush=True); t0 = time.time()
        try:
            fit = femu.lm_fit(pool, form, p0, max_iter=40)
        except RuntimeError as e:
            print("   FAILED:", e); out["fits"][name] = {"failed": str(e), "start": list(p0)}; continue
        fit["start"] = list(p0); fit["time_s"] = time.time() - t0
        out["fits"][name] = fit
        print(f"   -> cost {fit['cost']:.5f} in {fit['time_s']:.0f}s p={np.round(fit['p'], 4).tolist()}", flush=True)
        json.dump(out, open(f"results/fits/{form}.json", "w"), indent=1)
    ok = {k: v for k, v in out["fits"].items() if "cost" in v}
    best = min(ok, key=lambda k: ok[k]["cost"])
    out["best"] = dict(ok[best], start_name=best)
    json.dump(out, open(f"results/fits/{form}.json", "w"), indent=1)
    pool.close()
