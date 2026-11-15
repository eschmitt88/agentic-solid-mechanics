import sys,time,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"])
P=snap.FoldProblem()
t=time.time(); c=P.state_features(p); print('t',time.time()-t)
t=time.time(); c=P.state_features(p); print('t',time.time()-t)
print('feat',c['feat'],'events',c['n_events'],'nsnap',c['nsnap'],'d_u',c['d_u'],'d_pk',c['d_pk'],c['pk_dFdD'],'Eend',c['E_endpoints'])
for f in c['snaps']: print({k:v for k,v in f.items() if k not in('z_s','z_l','dds_dp','dloss_dp')}, np.round(f['dds_dp'],3))
G=snap.GridProblem(S=P.S); g=G.evaluate(p); print('grid feat',g['feat'])
print('dfeat fold\n',np.round(c['dfeat'],5)); print('dfeat grid\n',np.round(g['dfeat'],5))
