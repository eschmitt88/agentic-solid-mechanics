"""Calibration data helpers for the Damage Mechanics Challenge (reads the unpacked LD csv files)."""
import numpy as np, glob, os
LD_DIR = os.environ.get("DMC_LD_DIR", "/tmp/dmc_work/LD")
BEAMS = {g: [f"{g}{i}" for i in (3, 4, 5, 6)] for g in ("HC", "HB", "HA", "H45")}


def load_ld(name):
    a = np.genfromtxt(os.path.join(LD_DIR, name + "_LD.csv"), delimiter=",")
    a = a[~np.isnan(a).any(1)]
    return a[:, 1], a[:, 0]          # displacement mm, load N


def smooth(P, n=25):
    k = np.ones(n) / n
    return np.convolve(np.pad(P, n // 2, mode="edge"), k, mode="valid")[: len(P)]


def summary(name):
    d, P = load_ld(name)
    Ps = smooth(P)
    i = int(np.argmax(Ps))
    lo = (np.arange(len(P)) < i)
    s = lo & (Ps > 0.3 * Ps[i]) & (Ps < 0.8 * Ps[i])
    k = np.polyfit(d[s], Ps[s], 1)
    return dict(name=name, P_peak=float(Ps[i]), P_peak_raw=float(P.max()), d_peak=float(d[i]), k_mid=float(k[0]),
                d_offset=float(-k[1] / k[0]))   # displacement-axis intercept of the mid-range tangent (toe)


if __name__ == "__main__":
    import json
    out = {}
    for g, names in BEAMS.items():
        S = [summary(n) for n in names]
        for s in S:
            print({k: (round(v, 4) if not isinstance(v, str) else v) for k, v in s.items()})
        P = np.array([s["P_peak"] for s in S]); dp = np.array([s["d_peak"] - s["d_offset"] for s in S]); k = np.array([s["k_mid"] for s in S])
        out[g] = dict(P_mean=P.mean(), P_std=P.std(ddof=1), P_min=P.min(), P_max=P.max(), k_mean=k.mean(), k_std=k.std(ddof=1),
                      dpeak_toe_corrected_mean=dp.mean(), dpeak_raw_mean=float(np.mean([s["d_peak"] for s in S])), tests=S)
        print(g, {kk: round(float(v), 3) for kk, v in out[g].items() if kk != "tests"})
    json.dump(out, open("results/dmc/calibration_data_summary.json", "w"), indent=1)
