"""Problem-grain (PG) mutation operators for the trial-14 pilot C (see run_pg.py).

A PG edit changes the LOW-FIDELITY OPTIMISATION PROBLEM that produced a design; the child is
the LF optimiser's answer to the edited problem (lf_passive.lf_optimise_passive with
lf.GEN0_SETTINGS), then encoded and PCA-projected like every other child.

Editable parameters (schema PARAMS; every value is a valid lf_optimise input):
  V0            [0.20, 0.60]   LF volume target                      (gen 0: 0.25-0.50)
  Lmin          [0.05, 0.30]   minimum relative bar width            (gen 0: 0.10/0.15/0.20)
  rmin          {1.5, 2, 2.5, 3} density-filter radius, coarse cells  (gen 0: 2.0; quantised because
                               every new value JIT-compiles a filter, cf. run_morelf.py)
  theta_jitter  [0, 0.5] rad   noise on the initial (principal-stress) orientation  (gen 0: 0)
  init_jitter   [0, 0.15]      noise on the initial widths                          (gen 0: 0)
  seed          int >= 0       RNG of the two jitters (no effect if both are 0)     (gen 0: 0)
  boxes         <= 2 passive rectangles {kind: solid|void, x0, y0, x1, y1} in mm, sides 5-40 mm,
                inside [0, 100] x [0, 50]                                            (gen 0: none)

RAND operator (rand_edit): k ~ U{1, 2, 3} distinct edit types drawn uniformly from
  V0 (+N(0, 0.05)), Lmin (+N(0, 0.04)), rmin (uniform over the other 3 values),
  theta_jitter (U(0, 0.5)), init_jitter (U(0, 0.15)), seed (new U{0..2^31}),
  add_box (if < 2 boxes: kind 50/50, centre uniform in the domain, w ~ U(5, 25), h ~ U(5, 20) mm),
  remove_box (if any box);
continuous values clipped to their ranges. If `seed` is drawn while both jitters are 0 after the
edit, theta_jitter ~ U(0, 0.5) is also set (otherwise the edit would be a no-op).

LLM operator (llm_edit): one `claude -p` call (pinned model) with the parent's de-homogenised
picture and numbers; reply = short rationale + ```json {"edit": {...}}```. Values are validated
(clipped to range, rmin snapped, invalid keys dropped and recorded); a timeout, error, missing /
invalid JSON or an empty / no-op edit falls back to rand_edit (fallback=True, reason logged).
"""
from __future__ import annotations

import base64, copy, io, json, os, re, subprocess, time

import numpy as np

# ----------------------------------------------------------------------------- schema
RANGES = dict(V0=(0.20, 0.60), Lmin=(0.05, 0.30), theta_jitter=(0.0, 0.5), init_jitter=(0.0, 0.15))
RMIN_VALUES = (1.5, 2.0, 2.5, 3.0)
MAX_BOXES, BOX_SIDE = 2, (5.0, 40.0)
LX, LY = 100.0, 50.0
KEYS = ("V0", "Lmin", "rmin", "theta_jitter", "init_jitter", "seed", "boxes")


def gen0_params(V0, Lmin):
    return dict(V0=float(V0), Lmin=float(Lmin), rmin=2.0, theta_jitter=0.0, init_jitter=0.0, seed=0, boxes=[])


def _clip_box(b):
    """dict -> valid box dict or None. Sides clipped to BOX_SIDE, box shifted inside the domain."""
    try:
        kind = str(b["kind"]).lower()
        x0, y0, x1, y1 = (float(b[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        return None
    if kind not in ("solid", "void") or not all(np.isfinite([x0, y0, x1, y1])):
        return None
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    out = {"kind": kind}
    for (a, c, L, n0, n1) in ((x0, x1, LX, "x0", "x1"), (y0, y1, LY, "y0", "y1")):
        w = float(np.clip(c - a, *BOX_SIDE)); w = min(w, L)
        cen = float(np.clip((a + c) / 2, w / 2, L - w / 2))
        out[n0], out[n1] = round(cen - w / 2, 2), round(cen + w / 2, 2)
    return out


def validate(parent, edit):
    """Apply `edit` (dict of new values) to `parent` params. -> (child, changed, notes)."""
    child = copy.deepcopy(parent); notes = []
    if not isinstance(edit, dict):
        return child, {}, ["edit is not an object"]
    for k, v in edit.items():
        if k not in KEYS:
            notes.append(f"unknown key {k}"); continue
        try:
            if k in RANGES:
                lo, hi = RANGES[k]; f = float(v)
                if not np.isfinite(f):
                    raise ValueError
                if f < lo or f > hi:
                    notes.append(f"{k} clipped {f:g}")
                child[k] = float(np.clip(f, lo, hi))
            elif k == "rmin":
                f = float(v); s = min(RMIN_VALUES, key=lambda r: abs(r - f))
                if s != f:
                    notes.append(f"rmin snapped {f:g}->{s:g}")
                child[k] = s
            elif k == "seed":
                child[k] = int(abs(int(v))) % 2**31
            elif k == "boxes":
                if not isinstance(v, list):
                    raise ValueError
                bs = [_clip_box(b) if isinstance(b, dict) else None for b in v]
                if any(b is None for b in bs):
                    notes.append("invalid box dropped")
                bs = [b for b in bs if b is not None]
                if len(bs) > MAX_BOXES:
                    notes.append(f"{len(bs)} boxes truncated to {MAX_BOXES}")
                child[k] = bs[:MAX_BOXES]
        except (TypeError, ValueError, OverflowError):
            notes.append(f"invalid value for {k}")
    changed = {k: [parent[k], child[k]] for k in KEYS if child[k] != parent[k]}
    return child, changed, notes


# ----------------------------------------------------------------------------- RAND
def rand_edit(rng, parent):
    p = parent
    types = ["V0", "Lmin", "rmin", "theta_jitter", "init_jitter", "seed"]
    if len(p["boxes"]) < MAX_BOXES:
        types.append("add_box")
    if p["boxes"]:
        types.append("remove_box")
    pick = rng.choice(types, int(rng.integers(1, 4)), replace=False)
    e = {}
    boxes = list(p["boxes"])
    for t in pick:
        if t == "V0":
            e["V0"] = p["V0"] + rng.normal(0, 0.05)
        elif t == "Lmin":
            e["Lmin"] = p["Lmin"] + rng.normal(0, 0.04)
        elif t == "rmin":
            e["rmin"] = float(rng.choice([r for r in RMIN_VALUES if r != p["rmin"]]))
        elif t == "theta_jitter":
            e["theta_jitter"] = rng.uniform(*RANGES["theta_jitter"])
        elif t == "init_jitter":
            e["init_jitter"] = rng.uniform(*RANGES["init_jitter"])
        elif t == "seed":
            e["seed"] = int(rng.integers(0, 2**31))
        elif t == "add_box":
            w, h = rng.uniform(5, 25), rng.uniform(5, 20)
            cx, cy = rng.uniform(0, LX), rng.uniform(0, LY)
            boxes.append(dict(kind=str(rng.choice(["solid", "void"])), x0=cx - w / 2, y0=cy - h / 2,
                              x1=cx + w / 2, y1=cy + h / 2))
        elif t == "remove_box":
            boxes.pop(int(rng.integers(len(boxes))))
    if "add_box" in pick or "remove_box" in pick:
        e["boxes"] = boxes
    if "seed" in pick and max(e.get("theta_jitter", p["theta_jitter"]), e.get("init_jitter", p["init_jitter"])) == 0:
        e["theta_jitter"] = rng.uniform(*RANGES["theta_jitter"])
    child, changed, notes = validate(p, e)
    return child, changed, dict(rand_types=[str(t) for t in pick], notes=notes)


# ----------------------------------------------------------------------------- picture
def render_png(s_clean, removed, mode_mag=None, scale=2):
    """Fine fields (200, 400), row 0 = y = 0. Black = material kept, light grey = solid fragments
    removed as disconnected, faint blue grid every 10 mm on void, red shading = local strain of the first buckling mode."""
    from PIL import Image
    ny, nx = s_clean.shape
    rgb = np.full((ny, nx, 3), 255.0)
    yy, xx = np.mgrid[0:ny, 0:nx]
    grid = ((xx % 40) == 0) | ((yy % 40) == 0)
    rgb[grid] = (215, 222, 240)
    rgb[removed] = (190, 190, 190)
    rgb[s_clean] = (0, 0, 0)
    if mode_mag is not None:
        a = 0.9 * np.clip(mode_mag, 0, 1)[..., None]                # mode_mag: 1 = 99th percentile
        red = np.array([235.0, 20, 20])
        rgb = np.where(s_clean[..., None], (1 - a) * rgb + a * red, rgb)
    img = Image.fromarray(rgb[::-1].astype(np.uint8))           # y up
    img = img.resize((nx * scale, ny * scale), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------- LLM
SYSTEM = ("You are a structural topology-optimisation expert acting as a mutation operator inside an "
          "evolutionary search. Answer with a rationale of at most 3 sentences, then exactly one ```json "
          "block. Do not use tools.")

PROBLEM = (
    "Problem: a 100 mm x 50 mm plane-stress girder (E = 1, nu = 0.3), x to the right, y up. Fixed (non-design) "
    "parts: a 2.5 mm solid deck along the top edge carrying a uniform downward line load (total 1); two bearing "
    "pads 5 mm wide x 2.5 mm tall at the bottom-left (pinned: x and y springs) and bottom-right (roller: y "
    "springs) corners; a service duct = circular void of radius 6 mm centred at (25, 22.5) with a solid liner "
    "ring (radius 6 to 8.5 mm). Everything else is design domain. A design is produced by a low-fidelity (LF) "
    "homogenisation topology optimisation on a coarse 40 x 20 grid (2.5 mm cells): each cell holds a rank-2 "
    "lattice (two orthogonal bar families, widths w1, w2 in [0, 1] of a 7.5 mm period, orientation theta) and "
    "the optimiser minimises LF compliance under a volume target. The LF result is de-homogenised into a fine "
    "400 x 200 bar pattern and evaluated at high fidelity (HF); solid fragments not connected to the deck are "
    "removed. Before HF evaluation every design is projected onto a PCA basis of the 480 initial designs, so "
    "fine detail is partly smoothed out.")

PARAM_DOC = (
    "Editable LF-problem parameters (your edit replaces the parent's problem; the LF optimiser is re-run from "
    "scratch):\n"
    "- V0 in [0.20, 0.60]: LF volume target. The HF volume fraction follows it (initial population: {vfmap}).\n"
    "- Lmin in [0.05, 0.30]: minimum bar width (fraction of the 7.5 mm period) wherever material exists; "
    "larger = fewer, thicker bars.\n"
    "- rmin in {{1.5, 2.0, 2.5, 3.0}}: density-filter radius in coarse cells (2.5 mm); larger = coarser "
    "solid/void layout, fewer large holes.\n"
    "- theta_jitter in [0, 0.5] rad: noise on the initial bar orientation (principal-stress directions of a "
    "solid girder); leads the optimiser to a different local optimum.\n"
    "- init_jitter in [0, 0.15]: noise on the initial bar widths.\n"
    "- seed (integer): RNG seed of the two jitters (no effect when both are 0).\n"
    "- boxes: list of at most 2 passive rectangles forced solid or void in the LF problem, "
    "{{\"kind\": \"solid\"|\"void\", \"x0\":, \"y0\":, \"x1\":, \"y1\":}} in mm, sides 5-40 mm. "
    "Giving \"boxes\" replaces the whole list ([] removes all).")


def _fmt_params(p):
    d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
    return json.dumps(d)


def build_prompt(obj, parent_params, nums, front, vfmap):
    goal = ("Objectives: minimise volume fraction vf AND minimise compliance C (stiffness), i.e. push the "
            "Pareto front of (vf, C) down and left." if obj == "stiff" else
            "Objectives: minimise volume fraction vf AND maximise the first linear buckling load factor BLF, "
            "subject to C not exceeding the initial population's compliance envelope at the same vf. The red "
            "shading on the picture is the local strain of the first buckling mode: the members that buckle first.")
    fr = "; ".join(f"vf {v:.3f}: " + (f"C {c:.4g}" if obj == "stiff" else f"BLF {b:.4g}") for v, c, b in front)
    num = f"vf = {nums['vf']:.4f}, C = {nums['compliance']:.5g}"
    if obj == "buck":
        num += f", BLF = {nums['blf']:.5g}"
    num += (f", disconnected solid fragments removed = {nums['n_frag']} "
            f"({100 * nums['frag_area']:.1f} % of the design area)")
    return (
        f"{PROBLEM}\n\n{goal}\n\nPicture (800 x 400 px, 1 px = 0.125 mm, top edge = y 50 mm): the parent design. "
        "Black = material, light grey = solid fragments removed as disconnected, faint blue grid lines every "
        f"10 mm.\n\nParent HF numbers: {num}.\nCurrent elite non-dominated front (sorted by vf): {fr}.\n"
        f"Parent's LF problem: {_fmt_params(parent_params)}\n\n{PARAM_DOC.format(vfmap=vfmap)}\n\n"
        "Task: propose ONE edit of the parent's LF problem whose child design is likely to improve the Pareto "
        "trade-off (a design better than the current front, or extending it). Give a rationale of at most 3 "
        "sentences referring to what you see, then one ```json block of the form "
        "{\"edit\": {<only the parameters you change>: <new value>}}.")


def _claude_env():
    return {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k == "AI_AGENT")}


def call_claude(png, text, model, timeout=120.0, cwd=os.path.expanduser("~/projects/.claude-p-cwd")):
    """One `claude -p` call -> dict(ok, text, latency_s, usage, cost_usd, error)."""
    msg = {"type": "user", "message": {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"type": "text", "text": text}]}}
    cmd = ["claude", "-p", "--model", model, "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--disallowedTools", "Bash Edit Write Read Glob Grep Task WebFetch WebSearch NotebookEdit",
           "--system-prompt", SYSTEM]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True, timeout=timeout,
                           cwd=cwd, env=_claude_env())
    except subprocess.TimeoutExpired:
        return dict(ok=False, error="timeout", latency_s=time.time() - t0)
    lat = time.time() - t0
    res = None
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            res = e
    if res is None:
        return dict(ok=False, error=f"no result event (rc {p.returncode}): {p.stdout[-300:]} {p.stderr[-300:]}",
                    latency_s=lat)
    u = res.get("usage") or {}
    usage = {k: u.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                   "cache_creation_input_tokens")}
    if res.get("is_error") or res.get("subtype") != "success":
        return dict(ok=False, error=f"result error: {str(res.get('result'))[:300]}", latency_s=lat, usage=usage)
    return dict(ok=True, text=res.get("result") or "", latency_s=lat, usage=usage, cost_usd=res.get("total_cost_usd"),
                api_ms=res.get("duration_api_ms"))


def parse_reply(text):
    """-> (edit dict | None, rationale, error)."""
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    if not blocks:
        return None, text.strip(), "no json block"
    rationale = text[:text.find("```json")].strip()
    try:
        d = json.loads(blocks[-1])
    except json.JSONDecodeError as ex:
        return None, rationale, f"invalid json: {ex}"
    if isinstance(d, dict) and isinstance(d.get("edit"), dict):
        d = d["edit"]
    if not isinstance(d, dict):
        return None, rationale, "json is not an object"
    return d, rationale, None


def llm_edit(rng, parent, png, prompt, model, timeout=120.0):
    """-> (child_params, changed, info). Falls back to rand_edit on any failure."""
    r = call_claude(png, prompt, model, timeout)
    info = dict(latency_s=r.get("latency_s"), usage=r.get("usage"), cost_usd=r.get("cost_usd"), fallback=False,
                reply=r.get("text"))
    err = r.get("error")
    if r["ok"]:
        edit, rationale, err = parse_reply(r["text"])
        info["rationale"] = rationale
        if edit is not None:
            child, changed, notes = validate(parent, edit)
            info["notes"] = notes; info["raw_edit"] = edit
            if changed:
                return child, changed, info
            err = "no-op edit"
    child, changed, rinfo = rand_edit(rng, parent)
    info.update(fallback=True, fallback_reason=err, **rinfo)
    return child, changed, info


# ----------------------------------------------------------------------------- worker tasks
def lf_task(params):
    """Pool task: run the LF optimiser for one PG problem (GEN0_SETTINGS + params)."""
    import lf, lf_passive
    p = dict(params)
    r = lf_passive.lf_optimise_passive(p["V0"], p["Lmin"], boxes=p["boxes"], seed=p["seed"], rmin=p["rmin"],
                                       theta_jitter=p["theta_jitter"], init_jitter=p["init_jitter"],
                                       **lf.GEN0_SETTINGS)
    return {k: r[k] for k in ("w1", "w2", "theta", "compliance_lf", "vol_lf", "n_iter", "wall_s",
                              "n_solid_box", "n_void_box")}


def _buckling_mode(prob, s):
    """|first buckling mode| per fine pixel (200, 400) for cleaned solid mask s (hf.Problem.evaluate
    with eigenvectors; same K, KG, stress filter, eigsh settings)."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla, pypardiso, hf
    Ee = np.where(s.ravel(), 1.0, hf.E_VOID)
    Ee[prob.pad_mask.ravel()] = prob.cfg.get("pad_E", 1.0)
    K = prob.assemble(Ee).tocsr()
    solver = pypardiso.PyPardisoSolver()
    try:
        solver.set_matrix_type(2)
        Kup = sp.triu(K, format="csr"); Kup.sort_indices()
        solver.factorize(Kup)

        def fsolve(b):
            solver.set_phase(33)
            return solver._call_pardiso(Kup, np.ascontiguousarray(b, dtype=np.float64))
        u = fsolve(prob.F)
        sig = Ee[:, None] * (u[prob.edof] @ (prob.D @ (prob.Bc / prob.h)).T)
        rf = prob.cfg.get("kg_filter_r", 0.0)
        if rf > 0:
            sig = prob._filter_stress(sig, s, rf)
        sx, sy, txy = (sig[:, k] * s.ravel() for k in range(3))
        kg = (sx[:, None] * prob.Mxx.ravel()[None] + sy[:, None] * prob.Myy.ravel()[None]
              + txy[:, None] * prob.Mxy.ravel()[None]).ravel()
        KG = sp.csr_matrix((kg, (prob.iK, prob.jK)), shape=K.shape).tocsr()
        Minv = sla.LinearOperator(K.shape, matvec=fsolve, dtype=float)
        mu, V = sla.eigsh(-KG, k=1, M=K, Minv=Minv, which="LA", tol=prob.cfg.get("eig_tol", 1e-6),
                          ncv=prob.cfg.get("eig_ncv", None), maxiter=5000)
        psi = V[:, 0]
    finally:
        solver.free_memory(everything=True)
    # mode strain magnitude per element (localises on the buckling members; raw |psi| is dominated
    # by the global deflection), solid only, normalised by its 99th percentile on solid
    eps = psi[prob.edof] @ (prob.Bc / prob.h).T
    mag = np.linalg.norm(eps, axis=1).reshape(s.shape) * s
    return mag / max(np.percentile(mag[s], 99), 1e-30)


def render_task(task):
    """Pool task: (Fd, buck) -> picture inputs + fragment stats of the default realisation."""
    import run_arm
    from scipy import ndimage
    Fd, buck = task
    prob = run_arm._W["prob"]
    w1, w2, th = run_arm.fields_from_features(Fd, "inv5")
    phi = run_arm._W["dehom"].dehomogenise(w1, w2, th)["phi"]
    s, connected = prob.clean(phi)
    raw = phi.astype(bool).copy(); raw[prob.solid_fixed] = True; raw[prob.void_fixed] = False
    lab, n = ndimage.label(raw)
    n_kept = len(np.unique(lab[s]))                          # components kept (s is a subset of raw)
    removed = raw & ~s
    mode = None
    if buck and connected:
        try:
            mode = _buckling_mode(prob, s)
        except Exception:                                       # picture without overlay
            mode = None
    return dict(s=s, removed=removed, mode=mode, n_frag=int(n - n_kept),
                frag_area=float(removed[prob.design].mean()), connected=connected)
