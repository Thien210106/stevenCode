import unittest
from hill_climbing import HillClimbingSolver


class TestHillClimbingRestarts(unittest.TestCase):
    def test_restarts_reproducible(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        s1 = HillClimbingSolver(dist, seed=123)
        p1, c1, h1 = s1.solve(max_iter=200, restarts=5, restart_strategy='random')

        s2 = HillClimbingSolver(dist, seed=123)
        p2, c2, h2 = s2.solve(max_iter=200, restarts=5, restart_strategy='random')

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_perturb_strategy_changes_start(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        s = HillClimbingSolver(dist, seed=42)
        p, c, h = s.solve(max_iter=50, restarts=3, restart_strategy='perturb', restart_perturb_prob=1.0)

        # history should contain entries from multiple runs
        self.assertGreaterEqual(len(h), 2)

    def test_max_no_improve_per_run_short_circuits(self):
        # uniform distance matrix => no improvement possible
        n = 5
        dist = [[0 if i == j else 1 for j in range(n)] for i in range(n)]

        s = HillClimbingSolver(dist, seed=1)
        p, c, h = s.solve(max_iter=1000, restarts=3, max_no_improve_per_run=1)

        # with quick stopping per run history should be short
        self.assertLessEqual(len(h), 1 + 3)  # initial entries per run only


if __name__ == '__main__':
    unittest.main()
