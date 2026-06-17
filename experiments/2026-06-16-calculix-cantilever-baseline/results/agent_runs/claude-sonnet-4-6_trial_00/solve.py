#!/usr/bin/env python3
"""
CalculiX C3D20R cantilever beam: mesh generation, solve, parse, converge, report.

Problem:
  L=1.0 m (x), b=0.05 m (y), h=0.1 m (z)
  Encastre at x=0; P=1000 N downward (-z) at x=L
  E=2.1e11 Pa, nu=0.3
"""

import subprocess, os, math, sys

HOME = os.path.expanduser("~")
ENV = {
    **os.environ,
    "PATH": HOME + "/.local/bin:" + os.environ.get("PATH", ""),
    "MAMBA_ROOT_PREFIX": HOME + "/projects/.micromamba",
}

Lx, Ly, Lz = 1.0, 0.05, 0.1
E, nu, P = 2.1e11, 0.3, 1000.0


def nid(i, j, k, nx, ny, nz):
    """1-based node ID in half-index grid (2nx+1)×(2ny+1)×(2nz+1)."""
    return k * (2*ny + 1) * (2*nx + 1) + j * (2*nx + 1) + i + 1


def write_inp(nx, ny, nz):
    """Write beam.inp for nx×ny×nz C3D20R mesh and return tip node list."""
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    lines = [f"** C3D20R cantilever  mesh={nx}x{ny}x{nz}"]

    # --- Nodes ---
    lines.append("*NODE,NSET=NALL")
    for k in range(2*nz + 1):
        for j in range(2*ny + 1):
            for i in range(2*nx + 1):
                n = nid(i, j, k, nx, ny, nz)
                lines.append(f"{n},{i*dx/2:.10f},{j*dy/2:.10f},{k*dz/2:.10f}")

    # --- Elements (C3D20R node ordering: 8 corners then 12 midside) ---
    lines.append("*ELEMENT,TYPE=C3D20R,ELSET=EALL")
    eid = 1
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                i0, j0, k0 = 2*ex, 2*ey, 2*ez
                N = lambda ii, jj, kk: nid(ii, jj, kk, nx, ny, nz)
                c = [
                    # corners (nodes 1-8)
                    N(i0,   j0,   k0),   N(i0+2, j0,   k0),
                    N(i0+2, j0+2, k0),   N(i0,   j0+2, k0),
                    N(i0,   j0,   k0+2), N(i0+2, j0,   k0+2),
                    N(i0+2, j0+2, k0+2), N(i0,   j0+2, k0+2),
                    # midside (nodes 9-20)
                    N(i0+1, j0,   k0),   N(i0+2, j0+1, k0),    # mid(1-2), mid(2-3)
                    N(i0+1, j0+2, k0),   N(i0,   j0+1, k0),    # mid(3-4), mid(4-1)
                    N(i0+1, j0,   k0+2), N(i0+2, j0+1, k0+2),  # mid(5-6), mid(6-7)
                    N(i0+1, j0+2, k0+2), N(i0,   j0+1, k0+2),  # mid(7-8), mid(8-5)
                    N(i0,   j0,   k0+1), N(i0+2, j0,   k0+1),  # mid(1-5), mid(2-6)
                    N(i0+2, j0+2, k0+1), N(i0,   j0+2, k0+1),  # mid(3-7), mid(4-8)
                ]
                # Trailing comma on line 1 signals continuation to CalculiX
                lines.append(f"{eid}," + ",".join(map(str, c[:8])) + ",")
                lines.append(",".join(map(str, c[8:])))
                eid += 1

    # --- Node sets ---
    # C3D20R is serendipity (not Lagrange): a grid point at (i,j,k) is an
    # actual mesh node only when at most ONE of {i%2,j%2,k%2} is odd.
    # Face-center positions (two odd indices) are never connected to elements,
    # so including them in CLOAD would silently lose that force fraction.
    # At the i=0 and i=2*nx faces i is even, so the filter is j%2+k%2<=1.
    def face_real(j, k):
        return (j % 2 + k % 2) <= 1

    fixed = [nid(0,    j, k, nx, ny, nz)
             for k in range(2*nz+1) for j in range(2*ny+1) if face_real(j, k)]
    tip   = [nid(2*nx, j, k, nx, ny, nz)
             for k in range(2*nz+1) for j in range(2*ny+1) if face_real(j, k)]

    for name, ids in [("FIXED", fixed), ("TIP", tip)]:
        lines.append(f"*NSET,NSET={name}")
        for s in range(0, len(ids), 8):
            lines.append(",".join(map(str, ids[s:s+8])))

    # --- Material and section ---
    lines += [
        "*MATERIAL,NAME=STEEL",
        "*ELASTIC",
        f"{E:.6e},{nu}",
        "*SOLID SECTION,ELSET=EALL,MATERIAL=STEEL",
    ]

    # --- Step ---
    fz = -P / len(tip)   # equal nodal forces; total = P
    lines += [
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
        "FIXED,1,3",   # encastre: fix U1, U2, U3 at x=0
        "*CLOAD",
    ]
    for n in tip:
        lines.append(f"{n},3,{fz:.10e}")

    lines += [
        "*NODE FILE", "U",
        "*EL FILE", "S",
        "*NODE PRINT,NSET=TIP", "U",
        "*EL PRINT,ELSET=EALL", "S",
        "*END STEP",
    ]

    with open("beam.inp", "w") as f:
        f.write("\n".join(lines) + "\n")

    return tip


def run_ccx():
    for ext in ("dat", "frd", "cvg", "sta"):
        try:
            os.remove(f"beam.{ext}")
        except FileNotFoundError:
            pass
    return subprocess.run(
        ["micromamba", "run", "-n", "solidmech", "ccx", "beam"],
        env=ENV, capture_output=True, text=True,
    )


def parse_dat():
    """Return (|U3|_tip, max_von_mises) from beam.dat, or (None, None)."""
    try:
        text = open("beam.dat").read()
    except FileNotFoundError:
        return None, None

    # ---- Tip U3 ----
    u3s = []
    in_u = False
    for line in text.splitlines():
        ll = line.lower()
        if "displacements" in ll and "tip" in ll:
            in_u = True
            continue
        if in_u:
            parts = line.split()
            if len(parts) == 4:
                try:
                    int(parts[0])
                    u3s.append(float(parts[3]))
                except (ValueError, IndexError):
                    pass
            elif "for set" in ll and u3s:
                in_u = False

    tip_u3 = max(abs(v) for v in u3s) if u3s else None

    # ---- Von Mises from S11..S23 at integration points ----
    vm_max = 0.0
    in_s = False
    for line in text.splitlines():
        ll = line.lower()
        if "stresses" in ll and "for set" in ll:
            in_s = True
            continue
        if in_s:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    int(parts[0]); int(parts[1])
                    s11,s22,s33,s12,s13,s23 = (float(x) for x in parts[2:8])
                    vm = math.sqrt(0.5*(
                        (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                        + 6*(s12**2 + s13**2 + s23**2)
                    ))
                    vm_max = max(vm_max, vm)
                except (ValueError, IndexError):
                    pass
            elif "for set" in ll and in_s and vm_max > 0:
                in_s = False

    return tip_u3, (vm_max if vm_max > 0 else None)


if __name__ == "__main__":
    I_beam = Ly * Lz**3 / 12
    eb = P * Lx**3 / (3 * E * I_beam)
    print(f"Euler-Bernoulli reference: δ = {eb:.6e} m  (I={I_beam:.4e} m⁴)")

    prev_u3 = None
    final_u3 = final_mises = None
    converged = False
    ccx_runs = 0

    # Mesh sequence: refine in all directions
    configs = [(4,1,2), (8,1,2), (16,2,4), (32,2,4)]

    for nx, ny, nz in configs:
        n_elem = nx * ny * nz
        print(f"\n--- Mesh {nx}×{ny}×{nz}  ({n_elem} elements) ---", flush=True)
        write_inp(nx, ny, nz)
        r = run_ccx()
        ccx_runs += 1

        if r.returncode != 0:
            print("CCX FAILED")
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-1000:])
            sys.exit(1)

        u3, mises = parse_dat()

        if u3 is None:
            print("Displacement parse failed. beam.dat content:")
            try:
                print(open("beam.dat").read()[:5000])
            except FileNotFoundError:
                print("(beam.dat not found)")
            sys.exit(1)

        err = (u3 - eb) / eb * 100
        print(f"  |U3|      = {u3:.6e} m  ({err:+.2f}% vs EB)")
        print(f"  max Mises = {mises:.4e} Pa" if mises else "  max Mises = (not parsed)")

        if prev_u3 is not None:
            chg = abs(u3 - prev_u3) / max(abs(prev_u3), 1e-30) * 100
            print(f"  mesh-to-mesh Δ = {chg:.3f}%")
            if chg < 0.5:
                print("  → CONVERGED ✓")
                final_u3, final_mises = u3, mises
                converged = True
                break

        prev_u3 = u3
        final_u3, final_mises = u3, mises

    if not converged:
        print("\nWarning: convergence criterion not met; reporting finest mesh.")

    print(f"\n{'='*50}")
    print(f"CCX runs          : {ccx_runs}")
    print(f"tip_deflection_m  : {final_u3:.8e}")
    print(f"max_von_mises_pa  : {final_mises:.6e}" if final_mises else "max_von_mises_pa  : N/A")
