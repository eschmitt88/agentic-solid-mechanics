#!/usr/bin/env python3
"""Generate a CalculiX C3D20R hex mesh for a cantilever beam."""
import sys

L, b, h = 1.0, 0.05, 0.10  # x, y, z

# elements through each direction
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 2
nz = int(sys.argv[3]) if len(sys.argv) > 3 else 4
fname = sys.argv[4] if len(sys.argv) > 4 else "beam.inp"

P = -1000.0  # downward, -z, total

# C3D20: 20-node hex. Build a grid of nodes including mid-edge nodes.
# Use a full quadratic node grid: dimensions (2*nx+1) x (2*ny+1) x (2*nz+1),
# but the centers of faces/cells are NOT nodes in a serendipity element.
# Strategy: create node positions on the (2nx+1,2ny+1,2nz+1) lattice but only
# keep nodes that belong to a C3D20 element (i.e. corners + edge midpoints).

NX, NY, NZ = 2*nx+1, 2*ny+1, 2*nz+1

def keep(i, j, k):
    # a lattice point is a valid serendipity node unless it is an interior
    # point of an edge-face-cell where >=2 indices are odd... actually for
    # C3D20 we keep a point if at most one of (i,j,k) is odd.
    odd = (i % 2) + (j % 2) + (k % 2)
    return odd <= 1

# assign node ids
node_id = {}
nid = 0
nodes = []
for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            if keep(i, j, k):
                nid += 1
                node_id[(i, j, k)] = nid
                x = L * i / (NX - 1)
                y = b * j / (NY - 1)
                z = h * k / (NZ - 1)
                nodes.append((nid, x, y, z))

# C3D20 connectivity order (Abaqus/CalculiX):
# corners 1-8, then mid-edge 9-20.
# Local corner coords (in 0..2 lattice offsets within element cell):
# node ordering per CalculiX manual for C3D20:
# 1:(0,0,0) 2:(2,0,0) 3:(2,2,0) 4:(0,2,0)
# 5:(0,0,2) 6:(2,0,2) 7:(2,2,2) 8:(0,2,2)
# 9:(1,0,0) 10:(2,1,0) 11:(1,2,0) 12:(0,1,0)
# 13:(1,0,2) 14:(2,1,2) 15:(1,2,2) 16:(0,1,2)
# 17:(0,0,1) 18:(2,0,1) 19:(2,2,1) 20:(0,2,1)
offsets = [
    (0,0,0),(2,0,0),(2,2,0),(0,2,0),
    (0,0,2),(2,0,2),(2,2,2),(0,2,2),
    (1,0,0),(2,1,0),(1,2,0),(0,1,0),
    (1,0,2),(2,1,2),(1,2,2),(0,1,2),
    (0,0,1),(2,0,1),(2,2,1),(0,2,1),
]

elements = []
eid = 0
for ke in range(nz):
    for je in range(ny):
        for ie in range(nx):
            base = (2*ie, 2*je, 2*ke)
            conn = []
            for (di, dj, dk) in offsets:
                conn.append(node_id[(base[0]+di, base[1]+dj, base[2]+dk)])
            eid += 1
            elements.append((eid, conn))

# node sets: fixed at x=0 (i==0), loaded at x=L (i==NX-1)
fixed = [nid for (i,j,k), nid in node_id.items() if i == 0]
tip = [nid for (i,j,k), nid in node_id.items() if i == NX-1]

with open(fname, "w") as f:
    f.write("*HEADING\n Cantilever beam C3D20R\n")
    f.write("*NODE\n")
    for (n, x, y, z) in nodes:
        f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n")
    f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
    for (e, conn) in elements:
        # write 15 entries per line max per Abaqus; split lines
        items = [str(e)] + [str(c) for c in conn]
        # CalculiX allows continuation lines; put up to ~15 per line
        line1 = ", ".join(items[:16])
        line2 = ", ".join(items[16:])
        f.write(line1 + ",\n")
        f.write(line2 + "\n")
    f.write("*NSET, NSET=NFIX\n")
    f.write(",\n".join(str(n) for n in fixed) + "\n")
    f.write("*NSET, NSET=NTIP\n")
    f.write(",\n".join(str(n) for n in tip) + "\n")
    f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n210e9, 0.3\n")
    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write("*BOUNDARY\nNFIX, 1, 3, 0.0\n")
    f.write("*STEP\n*STATIC\n")
    # distribute total load P over tip nodes
    fpn = P / len(tip)
    f.write("*CLOAD\n")
    for n in tip:
        f.write(f"{n}, 3, {fpn:.10g}\n")
    f.write("*NODE PRINT, NSET=NTIP\nU\n")
    f.write("*EL PRINT, ELSET=EALL\nS\n")
    f.write("*NODE FILE\nU\n*EL FILE\nS\n")
    f.write("*END STEP\n")

print(f"wrote {fname}: {len(nodes)} nodes, {len(elements)} elements ({nx}x{ny}x{nz})")
print(f"tip nodes: {len(tip)}, fixed nodes: {len(fixed)}")
