import sys,time
sys.path.insert(0,'.')
import numpy as np
import snap
P=snap.FoldProblem(log=open('/dev/null','w'))
p=snap.snap_starts()['cos_h5_t1.6']
P.state_features(p)
t=time.time(); P.state_features(p+1e-3); t1=time.time()-t
G=snap.GridProblem(S=P.S,log=open('/dev/null','w')); G.evaluate(p)
t=time.time(); G.evaluate(p+1e-3); print('fold',round(t1,1),'grid',round(time.time()-t,1))
