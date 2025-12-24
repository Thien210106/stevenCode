import threading
import math

from hill_climbing import HillClimbingSolver
from pso import PSOSolver


def make_simple_matrix(n=4):
    # simple symmetric unit-weight complete graph (no self-loops)
    mat = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    return mat


def test_hill_climbing_respects_stop_event():
    mat = make_simple_matrix()
    ev = threading.Event()
    ev.set()  # already requested stop

    solver = HillClimbingSolver(mat, seed=1)
    best_path, best_cost, history = solver.solve(max_iter=1000, restarts=1, stop_event=ev)

    # with stop requested before starting, we expect no runs to have been performed
    assert isinstance(best_path, list)
    assert len(history) == 0


def test_pso_respects_stop_event():
    mat = make_simple_matrix()
    ev = threading.Event()
    ev.set()

    solver = PSOSolver(mat, num_particles=5, seed=1)
    best_path, best_cost, history = solver.solve(max_iter=100, patience=10, stop_event=ev)

    assert isinstance(best_path, list)
    assert math.isfinite(best_cost)
    assert len(history) >= 1
