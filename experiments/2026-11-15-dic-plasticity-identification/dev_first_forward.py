import sys,time
import numpy as np, specimen, fem
t0=time.time(); S=specimen.Specimen('O1-3'); print('build %.1fs'%(time.time()-t0))
print('E_dic st40 rms', np.sqrt((S.E_dic[2]**2).mean(1)), 'max Eyy', S.E_dic[2][1].max(), 'nnb min/med', S.nnb.min(), np.median(S.nnb))
# compare own strains to VIC strains
d=S.d
for i in range(3):
    v=np.stack([d['vic_Exx'][i][S.keep], d['vic_Eyy'][i][S.keep], d['vic_Exy'][i][S.keep]])
    print('own vs VIC relL2', np.sqrt(((S.E_dic[i]-v)**2).sum()/(v**2).sum()))
p=np.array([300.,900.,3.,300.,0,1,1,.5,3])
t0=time.time(); out=S.simulate(p); print('forward %.1fs'%(time.time()-t0), out['stats'])
print(np.round(out['F'][::5]/1e3,2)); print(np.round(S.F_meas[::5]/1e3,2)); print(S.metrics(out))
t0=time.time(); out=S.simulate(p,sens_idx=[0,1,2,3]); print('sens %.1fs'%(time.time()-t0))
