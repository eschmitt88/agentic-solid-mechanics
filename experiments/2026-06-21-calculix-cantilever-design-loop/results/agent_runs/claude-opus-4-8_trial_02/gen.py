import sys

L=1.0
def gen(b,h,fname,nx=20,ny=2,nz=4,P=2000.0):
    # serendipity quadratic grid: indices 0..2n, keep nodes with <=1 odd index
    nodes={}  # (i,j,k)->id
    coords={}
    nid=0
    def keep(i,j,k):
        return (i%2)+(j%2)+(k%2)<=1
    for k in range(2*nz+1):
        for j in range(2*ny+1):
            for i in range(2*nx+1):
                if keep(i,j,k):
                    nid+=1
                    nodes[(i,j,k)]=nid
                    x=i/(2*nx)*L
                    y=-b/2+j/(2*ny)*b
                    z=-h/2+k/(2*nz)*h
                    coords[nid]=(x,y,z)
    # element connectivity offsets (C3D20 ordering)
    off=[(0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2),
         (1,0,0),(2,1,0),(1,2,0),(0,1,0),(1,0,2),(2,1,2),(1,2,2),(0,1,2),
         (0,0,1),(2,0,1),(2,2,1),(0,2,1)]
    elems=[]
    eid=0
    for K in range(nz):
        for J in range(ny):
            for I in range(nx):
                eid+=1
                conn=[nodes[(2*I+dx,2*J+dy,2*K+dz)] for (dx,dy,dz) in off]
                elems.append((eid,conn))
    fix=[nid for (i,j,k),nid in nodes.items() if i==0]
    tip=[nid for (i,j,k),nid in nodes.items() if i==2*nx]
    f=P/len(tip)
    with open(fname,'w') as o:
        o.write("*NODE, NSET=NALL\n")
        for n in sorted(coords):
            x,y,z=coords[n]; o.write(f"{n}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        o.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
        for eid,conn in elems:
            row=[str(eid)]+[str(c) for c in conn]
            o.write(", ".join(row[:16])+",\n"+", ".join(row[16:])+"\n")
        o.write("*NSET, NSET=NFIX\n"+",\n".join(str(n) for n in sorted(fix))+"\n")
        o.write("*NSET, NSET=NTIP\n"+",\n".join(str(n) for n in sorted(tip))+"\n")
        o.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n2.1e11, 0.3\n*DENSITY\n7850.0\n")
        o.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
        o.write("*STEP\n*STATIC\n")
        o.write("*BOUNDARY\nNFIX,1,3\n")
        o.write("*CLOAD\n")
        for n in sorted(tip): o.write(f"{n}, 3, {-f:.8e}\n")
        o.write("*NODE PRINT, NSET=NTIP\nU\n")
        o.write("*END STEP\n")
    return len(coords),len(elems),len(tip)

if __name__=="__main__":
    b=float(sys.argv[1]); h=float(sys.argv[2])
    nn,ne,nt=gen(b,h,"beam.inp")
    print(f"b={b} h={h} nodes={nn} elems={ne} tipnodes={nt} mass={7850*L*b*h:.4f}")
