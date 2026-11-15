import sys,json,time
sys.path.insert(0,'.')
import numpy as np
import ccx, design
p=np.array(json.load(open('results/designs/T2_nominal.json'))['p'])
t=time.time(); d,F,info=ccx.run("test_3d",p,4,dmax=1.25,ninc=10,lc_div=2.5,depth3d=10.0,threads=2,keep=True); print(info,time.time()-t); print(np.round(F,5))
t=time.time(); d,F2,info=ccx.run("test_2d",p,4,dmax=1.25,ninc=10,lc_div=2.5,threads=2); print(info,time.time()-t); print(np.round(F2,5)); print('ratio 3D/PE',np.round(F/F2,4))
t=time.time(); d,F3,info=ccx.run("test_ef",p,4,dmax=1.25,ninc=10,lc_div=2.5,elastic_frame=True,threads=2,keep=True); print(info,time.time()-t); print(np.round(F3,5)); print('ratio EF/rigid',np.round(F3/F2,4))
