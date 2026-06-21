#!/usr/bin/env python3
"""Generate a C3D20R structured-hex CalculiX deck for a cantilever beam.

Cantilever along x in [0,L]; clamp at x=0 (all DOF). Rectangular section
b (y) x h (z), centred on y=z=0. Tip load total P downward (-z) at x=L,
distributed equally over the free-end face nodes.

C3D20R = 20-node quadratic hex, reduced integration -> good in bending,
no shear locking. We build a grid of nx*ny*nz quadratic elements using
the standard 3x3x3 corner+edge node layout per element (no centre nodes).
"""
import sys

def gen(b, h, L=1.0, nx=20, ny=2, nz=4, fname="beam.inp",
        E=210e9, nu=0.30, rho=7850.0, P=2000.0):
    # Quadratic element grid: node positions at half-integer spacing.
    # Number of node "stations" along each axis = 2*n + 1.
    Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1

    def xc(i): return L * i / (Nx-1)
    def yc(j): return -b/2 + b * j / (Ny-1)
    def zc(k): return -h/2 + h * k / (Nz-1)

    # A quadratic (serendipity) hex has nodes at all corners+edge-midpoints,
    # i.e. every (i,j,k) station EXCEPT those with >=2 of {i,j,k} at an
    # interior (odd) position. Build node ids only for serendipity stations.
    def is_node(i, j, k):
        odd = (i % 2) + (j % 2) + (k % 2)
        return odd <= 1  # corners (0 odd) and edge midpoints (1 odd)

    nid = {}
    nodes = []
    counter = 1
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if is_node(i, j, k):
                    nid[(i, j, k)] = counter
                    nodes.append((counter, xc(i), yc(j), zc(k)))
                    counter += 1

    # C3D20 node ordering (CalculiX/Abaqus):
    # corners 1-8 then mid-edge 9-20.
    # Local corner coords (in element, i.e. station offsets *2):
    # n1(0,0,0) n2(2,0,0) n3(2,2,0) n4(0,2,0)  bottom z=0
    # n5(0,0,2) n6(2,0,2) n7(2,2,2) n8(0,2,2)  top z=2
    # mid-edges:
    # 9 (1,0,0) 10(2,1,0) 11(1,2,0) 12(0,1,0)  bottom
    # 13(1,0,2) 14(2,1,2) 15(1,2,2) 16(0,1,2)  top
    # 17(0,0,1) 18(2,0,1) 19(2,2,1) 20(0,2,1)  verticals
    offs = [
        (0,0,0),(2,0,0),(2,2,0),(0,2,0),
        (0,0,2),(2,0,2),(2,2,2),(0,2,2),
        (1,0,0),(2,1,0),(1,2,0),(0,1,0),
        (1,0,2),(2,1,2),(1,2,2),(0,1,2),
        (0,0,1),(2,0,1),(2,2,1),(0,2,1),
    ]

    elems = []
    eid = 1
    for ek in range(nz):
        for ej in range(ny):
            for ei in range(nx):
                bi, bj, bk = 2*ei, 2*ej, 2*ek
                conn = []
                for (di, dj, dk) in offs:
                    conn.append(nid[(bi+di, bj+dj, bk+dk)])
                elems.append((eid, conn))
                eid += 1

    # Node sets
    clamp = [n for (i, j, k), n in nid.items() if i == 0]
    tip   = [(j, k, n) for (i, j, k), n in nid.items() if i == Nx-1]
    tip_nodes = [n for (_, _, n) in tip]

    with open(fname, "w") as f:
        f.write("*HEADING\n cantilever b=%.4f h=%.4f C3D20R\n" % (b, h))
        f.write("*NODE\n")
        for (n, x, y, z) in nodes:
            f.write("%d, %.10e, %.10e, %.10e\n" % (n, x, y, z))
        f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
        for (e, conn) in elems:
            # 20 nodes per element; split across lines (max 16 entries/line)
            ids = [str(e)] + [str(c) for c in conn]
            line1 = ", ".join(ids[:16])
            line2 = ", ".join(ids[16:])
            f.write(line1 + ",\n")
            f.write(line2 + "\n")
        f.write("*NSET, NSET=NCLAMP\n")
        f.write(",\n".join("%d" % n for n in clamp) + "\n")
        f.write("*NSET, NSET=NTIP\n")
        f.write(",\n".join("%d" % n for n in tip_nodes) + "\n")
        f.write("*MATERIAL, NAME=STEEL\n")
        f.write("*ELASTIC\n%.6e, %.3f\n" % (E, nu))
        f.write("*DENSITY\n%.6e\n" % rho)
        f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nNCLAMP, 1, 3, 0.0\n")
        # distribute P over tip nodes equally in -z (DOF 3)
        fpn = -P / len(tip_nodes)
        f.write("*CLOAD\n")
        for n in tip_nodes:
            f.write("%d, 3, %.10e\n" % (n, fpn))
        f.write("*NODE PRINT, NSET=NTIP\nU\n")
        f.write("*END STEP\n")

    return len(tip_nodes)

if __name__ == "__main__":
    b = float(sys.argv[1]); h = float(sys.argv[2])
    fname = sys.argv[3] if len(sys.argv) > 3 else "beam.inp"
    ntip = gen(b, h, fname=fname)
    print("wrote", fname, "tip nodes:", ntip)
