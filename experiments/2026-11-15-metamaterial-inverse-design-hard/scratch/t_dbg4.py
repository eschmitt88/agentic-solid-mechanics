import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt3_stalled.json"))["h5.5_t2.0_a2"]["p"])
P=snap.FoldProblem(log=sys.stdout)
c=P.state_features(p); f=c['snaps'][0]
print('base',c['feat'],f['d_s'],f['lam_pre'],f['bracket'])
for j,h in [(7,1e-4),(7,1e-5),(7,-1e-5),(5,1e-5),(5,-1e-5)]:
    q=p.copy(); q[j]+=h
    try:
        cq=P.state_features(q)
        print(j,h,'feat',cq['feat'],'nsnap',cq['nsnap'],'events',cq.get('n_events'),'finiteF',np.all(np.isfinite(cq['F'])), [ (s['d_s'],s['exact'],s.get('error')) for s in cq.get('snaps',[])])
    except Exception as e:
        import traceback; traceback.print_exc()
    X=P.S.coords(jnp.asarray(q))
    back=max(2e-4,4*abs(f['dds_dp'][j]*h))
    S=P.S
    d0=f['d_s']-back; z0=f['z_s']
    for dd in np.linspace(f['d_s'],d0,6)[1:]:
        z0,info=S.minimise(z0,float(dd),X,tol=1e-11)
        print('   walk',dd,info['converged'],info['iters'],info['shifted'],float(jnp.max(jnp.abs(z0-f['z_s']))))
    print('  refold',P.folds.refold(q,X,f,back), 'pred', f['d_s']+f['dds_dp'][j]*h)
