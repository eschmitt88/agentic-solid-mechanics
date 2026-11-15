import sys, time, cProfile, pstats
import numpy as np, specimen, fem
S = specimen.Specimen('O1-3')
p = np.array([330., 370., 5.3, 880., 0, 1, 1, .5, 3])
S.simulate(p, last_station=3)  # JIT warm
pr = cProfile.Profile(); pr.enable()
t0 = time.time(); out = S.simulate(p, sens_idx=[0, 1, 2, 3], last_station=20); print('time', time.time() - t0, out['stats'])
pr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(22)
