"""Grade rung 1 against the pre-registered criteria (PREREGISTRATION.md). Writes results/rung1_grade.json/.md."""
import json
M = json.load(open("results/pf_run_metrics.json"))


def finest(prefix):
    for n in (16, 8, 4, 2):
        if f"{prefix}_h{n}" in M:
            nxt = next((m for m in (n // 2, n // 4) if m >= 2 and f"{prefix}_h{m}" in M), None)
            return f"{prefix}_h{n}", (f"{prefix}_h{nxt}" if nxt else None)
    return None, None

rows = []
def add(cid, name, value, ref, tol_txt, ok, unit="", note=""):
    rows.append(dict(id=cid, quantity=name, value=value, reference=ref, tolerance=tol_txt, passed=bool(ok), unit=unit, note=note))

def pct(v, r):
    return 100 * (v / r - 1)

conv = {}
for prefix, pk, pu, ids, lab in (("t_l015", 0.734, 5.65e-3, ("T1", "T2"), "Tension l=0.015"), ("t_l0075", 0.720, 5.28e-3, ("T3", "T4"), "Tension l=0.0075"),
                                 ("s_l0075", 0.598, 0.0109, ("S1", "S2"), "Shear l=0.0075")):
    f, c = finest(prefix)
    m = M[f]
    add(ids[0], f"{lab} peak load", round(m["peak_F"], 4), pk, "±10 %", abs(pct(m["peak_F"], pk)) <= 10, "kN", f"{pct(m['peak_F'], pk):+.1f} % ({f})")
    add(ids[1], f"{lab} displacement at peak", round(m["u_peak"], 6), pu, "±10 %", abs(pct(m["u_peak"], pu)) <= 10, "mm", f"{pct(m['u_peak'], pu):+.1f} % ({f})")
    if c:
        conv[prefix] = dict(finest=f, coarser=c, change_pct=100 * abs(M[c]["peak_F"] / m["peak_F"] - 1))
f, _ = finest("t_l0075"); m = M[f]
add("T5", "Tension crack path: max deviation from y=0.5", round(m["path_max_dev"], 5), 0.0, "≤ l = 0.0075 mm, reaches right edge",
    m["path_max_dev"] <= 0.0075 + 1e-12 and m["crack_reaches_right"], "mm", f)
# crack path from the finest run that went to the end of the load programme (the h=l/8 run stops at u=0.0125)
fp = next(t for t in ("s_l0075_h4", "s_l0075_h2") if t in M and M[t].get("crack_end_x"))
m = M[fp]
add("S3", "Shear initial crack angle (fit 0.02–0.10 mm from tip)", round(m["crack_angle_deg"], 1), 61.0, "±10°", abs(m["crack_angle_deg"] - 61) <= 10, "deg",
    f"{fp}; other fit windows: {m.get('crack_angle_deg_w2', float('nan')):.1f}°, {m.get('crack_angle_deg_w3', float('nan')):.1f}°")
ok4 = m["crack_end_x"] is not None and abs(m["crack_end_x"] - 0.85) <= 0.07 and not m["second_crack_upper"] and not m["second_crack_left"]
add("S4", "Shear crack meets bottom edge at x", round(m["crack_end_x"], 3), 0.85, "±0.07 mm, single crack", ok4, "mm", fp)
for prefix, cc in conv.items():
    add("C1", f"Mesh convergence {prefix}: change of peak load between {cc['coarser']} and {cc['finest']}", round(cc["change_pct"], 2), 0.0, "≤ 3 %", cc["change_pct"] <= 3.0, "%")
sec = {}
for t, refs in (("s_l015", (0.54, 0.531)),):
    f, c = finest(t)
    sec[t] = dict(run=f, peak=M[f]["peak_F"], u_peak=M[f]["u_peak"], refs=refs)
out = dict(rows=rows, secondary=sec, all_pass=all(r["passed"] for r in rows))
json.dump(out, open("results/rung1_grade.json", "w"), indent=1)
with open("results/rung1_grade.md", "w") as fh:
    fh.write("| # | Quantity | This work | Reference | Tolerance | Result | Note |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['id']} | {r['quantity']} | {r['value']} {r['unit']} | {r['reference']} | {r['tolerance']} | {'PASS' if r['passed'] else 'FAIL'} | {r['note']} |\n")
print(open("results/rung1_grade.md").read()); print(sec)
