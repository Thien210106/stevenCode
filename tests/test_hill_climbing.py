import unittest
from hill_climbing import HillClimbingSolver


class TestHillClimbing(unittest.TestCase):
    def test_reproducibility(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        s1 = HillClimbingSolver(dist, seed=42)
        p1, c1, h1 = s1.solve(max_iter=200)

        s2 = HillClimbingSolver(dist, seed=42)
        p2, c2, h2 = s2.solve(max_iter=200)

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_start_path_and_two_opt(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        start = [3, 2, 1, 0]
        s = HillClimbingSolver(dist, seed=7)
        p, c, h = s.solve(max_iter=100, start_path=start, use_two_opt=True)

        self.assertEqual(sorted(p), [0, 1, 2, 3])

    def test_validation(self):
        bad = [[0, 1], [1, 0, 2]]
        with self.assertRaises(ValueError):
            HillClimbingSolver(bad)

        bad2 = [[0, -1], [-1, 0]]
        with self.assertRaises(ValueError):
            HillClimbingSolver(bad2)


if __name__ == '__main__':
    unittest.main()
