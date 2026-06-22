#!/usr/bin/env python3
"""Generate a CalculiX C3D20R mesh for a linearly-tapered cantilever.

Geometry: length L along x (clamp at x=0, free at x=L), constant width b along y,
height h(x) = h_root + (h_tip-h_root)*x/L along z, centred on z=0 so the deck
top/bottom surfaces taper symmetrically.

C3D20R = 20-node quadratic reduced-integration hex (no shear locking, no
hourglassing issues for bending; avoids the fully-integrated linear-hex C3D8
shear-locking trap the task warns about).

Clamp: all DOF fixed on the x=0 face (encastre).
Load: total P downward (-z) applied to the free face x=L, distributed over its
nodes by consistent-ish nodal lumping (equal split is adequate; we read
deflection, not the load face stress).
"""
import sys, numpy as np

def gen(inp_path, b, h_root, h_tip, L=1.0, nx=40, ny=2, nz=6, P=2000.0):
    E, nu, rho = 210e9, 0.30, 7850.0
    # Quadratic elements -> 2 nodes per element edge -> node grid is (2*nx+1) etc.
    Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1
    xs = np.linspace(0.0, L, Nx)
    ys = np.linspace(-b/2, b/2, Ny)
    zeta = np.linspace(-0.5, 0.5, Nz)  # normalised through-thickness

    def hx(x): return h_root + (h_tip - h_root) * (x / L)

    # Build full structured node grid first, then keep only nodes used by C3D20
    # (the quadratic serendipity hex omits face/body centre nodes).
    nid = -np.ones((Nx, Ny, Nz), dtype=int)
    coords = {}
    # Determine which (i,j,k) are serendipity nodes of at least one element.
    used = np.zeros((Nx, Ny, Nz), dtype=bool)
    # element corner index step is 2
    for ei in range(nx):
        for ej in range(ny):
            for ek in range(nz):
                i0, j0, k0 = 2*ei, 2*ej, 2*ek
                # 8 corners
                corners = [(i0,j0,k0),(i0+2,j0,k0),(i0+2,j0+2,k0),(i0,j0+2,k0),
                           (i0,j0,k0+2),(i0+2,j0,k0+2),(i0+2,j0+2,k0+2),(i0,j0+2,k0+2)]
                # 12 mid-edge nodes
                mids = [(i0+1,j0,k0),(i0+2,j0+1,k0),(i0+1,j0+2,k0),(i0,j0+1,k0),
                        (i0+1,j0,k0+2),(i0+2,j0+1,k0+2),(i0+1,j0+2,k0+2),(i0,j0+1,k0+2),
                        (i0,j0,k0+1),(i0+2,j0,k0+1),(i0+2,j0+2,k0+1),(i0,j0+2,k0+1)]
                for (i,j,k) in corners+mids:
                    used[i,j,k]=True

    counter = 1
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if not used[i,j,k]: continue
                x = xs[i]; y = ys[j]
                z = zeta[k]*hx(x)   # height scales locally -> true taper
                nid[i,j,k]=counter
                coords[counter]=(x,y,z)
                counter+=1

    elems=[]
    for ei in range(nx):
        for ej in range(ny):
            for ek in range(nz):
                i0,j0,k0=2*ei,2*ej,2*ek
                c=[(i0,j0,k0),(i0+2,j0,k0),(i0+2,j0+2,k0),(i0,j0+2,k0),
                   (i0,j0,k0+2),(i0+2,j0,k0+2),(i0+2,j0+2,k0+2),(i0,j0+2,k0+2)]
                m=[(i0+1,j0,k0),(i0+2,j0+1,k0),(i0+1,j0+2,k0),(i0,j0+1,k0),
                   (i0+1,j0,k0+2),(i0+2,j0+1,k0+2),(i0+1,j0+2,k0+2),(i0,j0+1,k0+2),
                   (i0,j0,k0+1),(i0+2,j0,k0+1),(i0+2,j0+2,k0+1),(i0,j0+2,k0+1)]
                conn=[nid[ii] for ii in c+m]
                elems.append(conn)

    # clamp nodes: x==0 face (i==0)
    clamp=[nid[0,j,k] for j in range(Ny) for k in range(Nz) if nid[0,j,k]>0]
    # tip nodes: x==L face (i==Nx-1)
    tip=[nid[Nx-1,j,k] for j in range(Ny) for k in range(Nz) if nid[Nx-1,j,k]>0]
    # tip reference node for deflection: centre of tip face (y=0,z=0)
    jc=(Ny-1)//2; kc=(Nz-1)//2
    tip_ref=nid[Nx-1,jc,kc]

    with open(inp_path,'w') as f:
        f.write("*HEADING\nTapered cantilever, C3D20R\n")
        f.write("*NODE\n")
        for n in sorted(coords):
            x,y,z=coords[n]; f.write(f"{n}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write("*ELEMENT, TYPE=C3D20R, ELSET=DECK\n")
        for e,conn in enumerate(elems,1):
            vals=[e]+conn  # 21 entries total; ccx allows max 16 per line
            # first line up to 15 nodes (16 entries incl elem id), rest continuation
            first=vals[:16]; rest=vals[16:]
            f.write(", ".join(str(v) for v in first)+",\n")
            f.write(", ".join(str(v) for v in rest)+"\n")
        f.write("*NSET, NSET=CLAMP\n")
        f.write(",\n".join(str(c) for c in clamp)+"\n")
        f.write("*NSET, NSET=TIP\n")
        f.write(",\n".join(str(c) for c in tip)+"\n")
        f.write(f"*NSET, NSET=TIPREF\n{tip_ref}\n")
        f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n")
        f.write(f"{E:.6e}, {nu}\n")
        f.write("*SOLID SECTION, ELSET=DECK, MATERIAL=STEEL\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nCLAMP, 1, 3, 0.0\n")
        # distribute P over tip nodes equally
        per=P/len(tip)
        f.write("*CLOAD\n")
        for t in tip:
            f.write(f"{t}, 3, {-per:.8e}\n")
        f.write("*NODE PRINT, NSET=TIPREF\nU\n")
        f.write("*END STEP\n")
    return tip_ref, len(elems), len(coords)

if __name__=="__main__":
    inp=sys.argv[1]; b=float(sys.argv[2]); hr=float(sys.argv[3]); ht=float(sys.argv[4])
    nx=int(sys.argv[5]) if len(sys.argv)>5 else 40
    nz=int(sys.argv[6]) if len(sys.argv)>6 else 6
    tr,ne,nn=gen(inp,b,hr,ht,nx=nx,nz=nz)
    print(f"wrote {inp}: tip_ref_node={tr} elems={ne} nodes={nn}")
