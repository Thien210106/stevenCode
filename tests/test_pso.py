import unittest
from pso import PSOSolver


class TestPSO(unittest.TestCase):
    def test_reproducibility(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        s1 = PSOSolver(dist, num_particles=10, seed=42)
        p1, c1, h1 = s1.solve(max_iter=50)

        s2 = PSOSolver(dist, num_particles=10, seed=42)
        p2, c2, h2 = s2.solve(max_iter=50)

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_early_stopping(self):
        # matrix where all off-diagonal distances are equal -> no improvement expected
        n = 5
        dist = [[0 if i == j else 1 for j in range(n)] for i in range(n)]

        s = PSOSolver(dist, num_particles=3, seed=1)
        best_path, best_cost, history = s.solve(max_iter=100, patience=1)

        # history should be short due to early stopping (initial + at most 1 iteration)
        # Note: hybrid HC may append additional entries in history; allow one extra
        self.assertLessEqual(len(history), 4)

    def test_move_towards_exact(self):
        dist = [[0, 1], [1, 0]]
        s = PSOSolver(dist, num_particles=2, seed=7)
        cur = [0, 1, 2, 3]
        target = [3, 2, 1, 0]

        out = s.move_towards(cur, target, prob=1.0)
        self.assertEqual(out, target)

    def test_validation_raises(self):
        # non-square matrix
        bad = [[0, 1], [1, 0, 2]]
        with self.assertRaises(ValueError):
            PSOSolver(bad)

        # negative distances
        bad2 = [[0, -1], [-1, 0]]
        with self.assertRaises(ValueError):
            PSOSolver(bad2)

    def test_move_operator_two_opt(self):
        from pso import PSOConfig

        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        cfg = PSOConfig(move_operator='2opt')
        s = PSOSolver(dist, num_particles=6, seed=5, config=cfg)
        best_path, best_cost, history = s.solve(max_iter=30, patience=5)

        # sanity checks
        self.assertIsInstance(best_cost, (int, float))
        self.assertEqual(sorted(best_path), [0, 1, 2, 3])

    def test_parallel_consistency(self):
        from pso import PSOConfig

        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        cfg1 = PSOConfig(num_workers=1)
        s1 = PSOSolver(dist, num_particles=6, seed=11, config=cfg1)
        p1, c1, h1 = s1.solve(max_iter=30, patience=5)

        cfg2 = PSOConfig(num_workers=2)
        s2 = PSOSolver(dist, num_particles=6, seed=11, config=cfg2)
        p2, c2, h2 = s2.solve(max_iter=30, patience=5)

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)


if __name__ == '__main__':
    unittest.main()
