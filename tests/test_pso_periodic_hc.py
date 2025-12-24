import unittest
from pso import PSOSolver, PSOConfig


class TestPSOPeriodicHC(unittest.TestCase):
    def test_periodic_hc_reproducible(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        cfg = PSOConfig(hc_use_solver=True, hc_interval=1, hc_restarts=1, hc_max_iter_per_run=20, hc_use_two_opt=True)
        s1 = PSOSolver(dist, num_particles=6, seed=999, config=cfg)
        p1, c1, h1 = s1.solve(max_iter=20)

        s2 = PSOSolver(dist, num_particles=6, seed=999, config=cfg)
        p2, c2, h2 = s2.solve(max_iter=20)

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_periodic_hc_improves_over_end_only(self):
        dist = [
            [0, 50, 9, 10],
            [50, 0, 6, 4],
            [9, 6, 0, 8],
            [10, 4, 8, 0],
        ]

        cfg_end = PSOConfig(hc_use_solver=True, hc_interval=0, hc_restarts=2, hc_max_iter_per_run=50, hc_use_two_opt=True)
        s_end = PSOSolver(dist, num_particles=10, seed=1234, config=cfg_end)
        p_end, c_end, h_end = s_end.solve(max_iter=40)

        cfg_period = PSOConfig(hc_use_solver=True, hc_interval=5, hc_restarts=2, hc_max_iter_per_run=50, hc_use_two_opt=True)
        s_period = PSOSolver(dist, num_particles=10, seed=1234, config=cfg_period)
        p_period, c_period, h_period = s_period.solve(max_iter=40)

        self.assertLessEqual(c_period, c_end)

    def test_hc_apply_to_topk(self):
        dist = [
            [0, 20, 9, 10],
            [20, 0, 6, 4],
            [9, 6, 0, 8],
            [10, 4, 8, 0],
        ]

        cfg = PSOConfig(hc_use_solver=True, hc_interval=2, hc_apply_to='topk', hc_top_k=2, hc_restarts=1, hc_max_iter_per_run=30)
        s = PSOSolver(dist, num_particles=8, seed=42, config=cfg)
        p, c, h = s.solve(max_iter=30)

        self.assertIsInstance(c, (int, float))
        self.assertEqual(sorted(p), [0,1,2,3])
