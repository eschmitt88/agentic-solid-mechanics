import statistics
u3=[]; cap=False
for l in open('beam.dat').read().splitlines():
    if 'displacements' in l.lower(): cap=True; continue
    if cap:
        p=l.split()
        if len(p)>=4:
            try: u3.append(float(p[3]))
            except: pass
        elif u3: cap=False
print(f"FE tip deflection avg = {statistics.mean(u3)*1000:.4f} mm   max = {min(u3)*1000:.4f} mm")
