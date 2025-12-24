import threading
from hill_climbing import HillClimbingSolver
from pso import PSOSolver

mat = [[0 if i == j else 1 for j in range(4)] for i in range(4)]
ev = threading.Event()
# test 1: immediate stop
ev.set()
hc = HillClimbingSolver(mat, seed=1)
res = hc.solve(max_iter=1000, restarts=1, stop_event=ev)
print('HC returned: path_len=', len(res[0]), 'history_len=', len(res[2]))

pso = PSOSolver(mat, num_particles=5, seed=1)
res2 = pso.solve(max_iter=10, patience=1, stop_event=ev)
print('PSO returned: path_len=', len(res2[0]), 'history_len=', len(res2[2]))

# test 2: set stop after a short timeout during run
import threading, time

ev2 = threading.Event()

import threading

def run_hc():
    hc2 = HillClimbingSolver(mat, seed=2)
    r = hc2.solve(max_iter=1000, restarts=0, stop_event=ev2)
    print('HC2 done: iterations in history=', len(r[2]))

t = threading.Thread(target=run_hc)
t.start()
# let it run a tiny bit
time.sleep(0.01)
ev2.set()
t.join()
print('HC2 thread finished')
