import sys,json,time
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import snap
P=snap.MarginProblem(log=open('/dev/null','w'))
p=snap.snap_starts()['hinge_h5_t1.5_b-0.4']
t=time.time(); c=P.state_features(p); print('rho',c['rho'],'at',c['d_rho'],time.time()-t)
t=time.time(); c=P.state_features(p); print('time 2nd',time.time()-t)
for j in (0,1,4,6):
    h=1e-5; pp=p.copy(); pp[j]+=h; pm=p.copy(); pm[j]-=h
    fd=(P.state_features(pp)['rho']-P.state_features(pm)['rho'])/(2*h)
    print(j,'drho analytic',c['drho'][j],'FD',fd,'rel',abs(fd/c['drho'][j]-1))
p2=np.array(json.load(open('results/designs/bistable.json'))['p']); c2=P.state_features(p2); print('AO success design rho',c2['rho'],'residual',max(0,1-c2['rho']/P.RHO_C))
