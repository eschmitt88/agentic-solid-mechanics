"""Pre-processing of a Sandia *-Data.mat file into the compact arrays the pre-registered pipeline uses.
One rule set for every specimen (see PREREGISTRATION.md).  Never writes into raw/."""
import os, sys, numpy as np, h5py
import scipy.io as sio

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw", "sandia-304l-dic"))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cache")
NST = 41
FIELD_ST = (13, 27, 40)
FORCE_ST0 = 5
GEOM = {"O1": "O1-NominalGeometry.mat", "O5": "O5-NominalGeometry.mat", "O14": "O14-NominalGeometry.mat",
        "O15": "O15-NominalGeometry.mat", "O18": "O18-NominalGeometry.mat", "X": "X-Specimen-NominalGeometry.mat",
        "D": "D-Specimen-NominalGeometry.mat", "DB": "TensileDogBone-NominalGeometry.mat"}


def load_geometry(key):
    f = os.path.join(RAW, "geometry", GEOM[key])
    try:
        m = sio.loadmat(f)
        bo = np.asarray(m["bOut"], float)
        bh = [np.asarray(h, float) for h in m["bHole"].ravel()] if "bHole" in m else []
    except NotImplementedError:
        h = h5py.File(f, "r")
        bo = np.array(h["bOut"]).T
        bh = [np.array(h[r]).T for r in np.array(h["bHole"]).ravel()] if "bHole" in h else []
    bh = [b for b in bh if b.ndim == 2 and b.shape[1] == 2 and len(b) > 2]
    return bo, bh


def geom_key(spec):
    if spec.startswith("X"):
        return "X"
    return spec.split("-")[0]


def fit_bc(X0, Y0, Ust, Vst, bad, ylo, yhi):
    """cubic fits of U, V on the two cut lines at every station"""
    coef = np.zeros((NST, 2, 2, 4)); bc_rms = np.zeros((NST, 2, 2)); xr = np.zeros((2, 2))
    for il, yc in enumerate((ylo, yhi)):
        band = np.abs(Y0 - yc) <= 0.75
        xr[il] = [X0[band].min(), X0[band].max()]
        for j in range(NST):
            ok = band & ~bad[j]
            xs = (X0[ok] - 30.0) / 30.0
            A = np.vander(xs, 4)
            for ic, D in enumerate((Ust, Vst)):
                c, *_ = np.linalg.lstsq(A, D[j, ok], rcond=None)
                coef[j, il, ic] = c
                bc_rms[j, il, ic] = np.sqrt(np.mean((A @ c - D[j, ok]) ** 2))
    return coef, bc_rms, xr


def prepare(spec, role, force=False):
    """role: 'calibration' or 'heldout' (sub-folder of raw/.../large)."""
    out = os.path.join(CACHE, f"{spec}.npz")
    if os.path.exists(out) and not force:
        return out
    os.makedirs(CACHE, exist_ok=True)
    h = h5py.File(os.path.join(RAW, "large", role, f"{spec}-Data.mat"), "r")
    t = np.array(h["time"]).ravel(); ext = np.array(h["extension"]).ravel(); F = np.array(h["force"]).ravel() * 1e3  # N
    iF = int(np.nanargmax(F)); xF = ext[iF]
    targets = np.arange(NST) / (NST - 1) * 0.9 * xF
    cand = np.arange(0, iF + 1); cand = cand[np.isfinite(ext[cand]) & np.isfinite(F[cand])]
    frames = np.array([cand[np.argmin(np.abs(ext[cand] - x))] for x in targets])
    # station 0 = the DIC reference state (zero displacement); keep the frame for the record
    rollmax = np.array([np.nanmax(F[max(0, i - 5): i + 1]) if np.isfinite(F[max(0, i - 5): i + 1]).any() else np.nan for i in range(len(F))])  # preceding 5 frames (Amendment 1)
    drop = F[frames] < rollmax[frames] - 0.01 * F[iF]
    X0 = np.array(h["X0"]).ravel(); Y0 = np.array(h["Y0"]).ravel()
    U = np.array(h["U"]); V = np.array(h["V"]); S = np.array(h["sigma"])
    Ust = U[frames]; Vst = V[frames]; Sst = S[frames]
    del U, V, S
    bad = np.isnan(Ust) | np.isnan(Vst) | (Sst < 0)
    ylo = np.nanmin(Y0) + 3.0; yhi = np.nanmax(Y0) - 3.0
    coef, bc_rms, xr = fit_bc(X0, Y0, Ust, Vst, bad, ylo, yhi)
    coef[0] = 0.0  # station 0 is the undeformed reference
    fs = list(FIELD_ST)
    valid = ~bad[fs].any(0)
    vic = {k: np.array(h[k])[frames[fs]].astype(np.float32) for k in ("Exx", "Eyy", "Exy")}
    Tm = None
    if "IR_temperature" in h:
        Tm = np.array([np.nanmean(h["IR_temperature"][0, int(fr), :]) for fr in frames[[1, 20, 40]]])
    np.savez_compressed(out, spec=spec, time=t, ext=ext, force=F, iF=iF, xF=xF, frames=frames, drop=drop,
                        X0=X0, Y0=Y0, U=Ust[fs], V=Vst[fs], valid=valid, ylo=ylo, yhi=yhi, coef=coef, bc_rms=bc_rms,
                        bc_xrange=xr, vic_Exx=vic["Exx"], vic_Eyy=vic["Eyy"], vic_Exy=vic["Exy"],
                        T_mean=np.array(Tm if Tm is not None else [np.nan] * 3), Fst=F[frames], ext_st=ext[frames])
    return out


if __name__ == "__main__":
    for s in sys.argv[2:]:
        o = prepare(s, sys.argv[1], force=True)
        d = np.load(o)
        print(s, "frames", d["frames"][[0, 1, 5, 13, 27, 40]], "xF %.2f" % d["xF"], "Fmax %.0f" % d["force"][d["iF"]],
              "drops at stations", np.where(d["drop"])[0], "valid frac %.4f" % d["valid"].mean(),
              "ylo/yhi %.2f %.2f" % (d["ylo"], d["yhi"]), "bc rms max (mm) %.4f" % d["bc_rms"].max(),
              "T", np.round(d["T_mean"], 1), "npts", len(d["X0"]))
