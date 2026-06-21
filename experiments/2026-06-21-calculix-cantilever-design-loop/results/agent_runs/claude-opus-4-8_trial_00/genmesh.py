#!/usr/bin/env python3
import sys

# Parametric C3D20R cantilever mesh generator.
# Args: b h nx ny nz
b = float(sys.argv[1]); h = float(sys.argv[2])
L = 1.0
nx = int(sys.argv[3]); ny = int(sys.argv[4]); nz = int(sys.argv[5])
E = 2.1e11; nu = 0.3; rho = 7850.0
P = 2000.0  # total downward load

# C3D20: 20-node hex. Build via a "quadratic" grid: full node grid at
# half-step resolution, then drop the cell-center / face-center nodes that
# C3D20 doesn't use (keep corners + edge midpoints only).
# Node grid indices i in 0..2nx, j 0..2ny, k 0..2nz (steps of half element).
Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1

def used(i, j, k):
    # number of indices that are odd (midside in that direction)
    odd = (i % 2) + (j % 2) + (k % 2)
    return odd <= 1  # corner(0) or edge-midside(1) only

nid = {}
nodes = []
counter = 1
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if not used(i, j, k):
                continue
            x = L * i / (2*nx)
            y = b * j / (2*ny)
            z = h * k / (2*nz)
            nid[(i, j, k)] = counter
            nodes.append((counter, x, y, z))
            counter += 1

# C3D20 connectivity order (Abaqus/CalculiX):
# corners of bottom (k), then top, then midside bottom, midside top, vertical mids
# Local corner offsets (in half-steps) for element (ex,ey,ez):
# base i0=2ex, j0=2ey, k0=2ez
def c20(ex, ey, ez):
    i0, j0, k0 = 2*ex, 2*ey, 2*ez
    # 8 corners
    corners = [
        (i0,   j0,   k0),   # 1
        (i0+2, j0,   k0),   # 2
        (i0+2, j0+2, k0),   # 3
        (i0,   j0+2, k0),   # 4
        (i0,   j0,   k0+2), # 5
        (i0+2, j0,   k0+2), # 6
        (i0+2, j0+2, k0+2), # 7
        (i0,   j0+2, k0+2), # 8
    ]
    # 12 midside: bottom face edges 9-12, top face edges 13-16, verticals 17-20
    mids = [
        (i0+1, j0,   k0),   # 9  (1-2)
        (i0+2, j0+1, k0),   # 10 (2-3)
        (i0+1, j0+2, k0),   # 11 (3-4)
        (i0,   j0+1, k0),   # 12 (4-1)
        (i0+1, j0,   k0+2), # 13 (5-6)
        (i0+2, j0+1, k0+2), # 14 (6-7)
        (i0+1, j0+2, k0+2), # 15 (7-8)
        (i0,   j0+1, k0+2), # 16 (8-5)
        (i0,   j0,   k0+1), # 17 (1-5)
        (i0+2, j0,   k0+1), # 18 (2-6)
        (i0+2, j0+2, k0+1), # 19 (3-7)
        (i0,   j0+2, k0+1), # 20 (4-8)
    ]
    return [nid[c] for c in corners] + [nid[m] for m in mids]

elems = []
eid = 1
for ez in range(nz):
    for ey in range(ny):
        for ex in range(nx):
            elems.append((eid, c20(ex, ey, ez)))
            eid += 1

# Node sets: encastre at x=0, load nodes at x=L
fixed = [n for (n, x, y, z) in nodes if abs(x) < 1e-9]
tip = [n for (n, x, y, z) in nodes if abs(x - L) < 1e-9]

# Distribute total load P over tip nodes. For a quadratic face, simple equal
# split is an approximation; fine for tip deflection. Use equal split.
fload = -P / len(tip)

with open("beam.inp", "w") as f:
    f.write("*NODE, NSET=NALL\n")
    for (n, x, y, z) in nodes:
        f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n")
    f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
    for (e, conn) in elems:
        items = [str(e)] + [str(c) for c in conn]  # 21 entries
        # max 16 entries per line; first line 16, continuation 5
        f.write(", ".join(items[:16]) + ",\n")
        f.write(", ".join(items[16:]) + "\n")
    f.write("*NSET, NSET=FIX\n")
    f.write(",\n".join(str(n) for n in fixed) + "\n")
    f.write("*NSET, NSET=TIP\n")
    f.write(",\n".join(str(n) for n in tip) + "\n")
    f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n")
    f.write(f"{E:.6g}, {nu}\n")
    f.write("*DENSITY\n")
    f.write(f"{rho}\n")
    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write("*STEP\n*STATIC\n")
    f.write("*BOUNDARY\nFIX, 1, 3, 0.0\n")
    f.write("*CLOAD\n")
    for n in tip:
        f.write(f"{n}, 3, {fload:.10g}\n")
    f.write("*NODE PRINT, NSET=TIP\nU\n")
    f.write("*EL PRINT, ELSET=EALL\nS\n")
    f.write("*END STEP\n")

mass = rho * L * b * h
print(f"b={b} h={h} mass={mass:.4f} kg  nodes={len(nodes)} elems={len(elems)} tipnodes={len(tip)}")
