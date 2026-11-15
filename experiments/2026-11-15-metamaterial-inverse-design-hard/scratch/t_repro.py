import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import snap, fem, design
r=json.load(open('results/snap/AM+O2@120__cos_h5_t1.6.json'))
p=np.array(r['p'])
h=[x for x in r['history'] if x['p']==r['p']][0]; print('arm feat in history',h['feat'],'rho',h.get('rho'),'nsnap',h['nsnap'])
print('score in file',r['score']['feat'])
P=snap.MarginProblem(log=open('/dev/null','w'))
for k in range(2):
    c=P.state_features(p); print('state_features',c['feat'],'rho',c['rho'],'d_u',c['d_u'],'nsnap',c['nsnap'],'events',c['n_events'])
    sc=snap.score(P.S,p); print('score',sc['feat'])
F=np.array(sc['F']); idx=snap.indices(F); print('idx',idx, 'F around i2', np.round(F[idx[2]-3:idx[2]+3]*1e3,3), 'zero crossings up', [i for i in range(1,len(F)) if F[i-1]<0<=F[i]], 'down',[i for i in range(1,len(F)) if F[i-1]>0>=F[i]])
