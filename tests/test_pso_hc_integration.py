import unittest
from pso import PSOSolver, PSOConfig
from hill_climbing import HillClimbingSolver


class TestPSOHillClimbIntegration(unittest.TestCase):
    def test_hc_refine_reproducible(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        cfg = PSOConfig(hc_use_solver=True, hc_restarts=2, hc_max_iter_per_run=50, hc_use_two_opt=True)
        s1 = PSOSolver(dist, num_particles=6, seed=11, config=cfg)
        p1, c1, h1 = s1.solve(max_iter=30)

        s2 = PSOSolver(dist, num_particles=6, seed=11, config=cfg)
        p2, c2, h2 = s2.solve(max_iter=30)

        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_hc_refine_improves_over_swap_refine(self):
        dist = [
            [0, 20, 9, 10],
            [20, 0, 6, 4],
            [9, 6, 0, 8],
            [10, 4, 8, 0],
        ]

        # PSO without HillClimbing refinement
        cfg_no_hc = PSOConfig(hc_use_solver=False)
        s_no_hc = PSOSolver(dist, num_particles=8, seed=123, config=cfg_no_hc)
        p_no_hc, c_no_hc, h_no_hc = s_no_hc.solve(max_iter=50)

        # PSO with HillClimbing refinement
        cfg_hc = PSOConfig(hc_use_solver=True, hc_restarts=3, hc_max_iter_per_run=100, hc_use_two_opt=True)
        s_hc = PSOSolver(dist, num_particles=8, seed=123, config=cfg_hc)
        p_hc, c_hc, h_hc = s_hc.solve(max_iter=50)

        # Hill-climbing should not worsen the solution
        self.assertLessEqual(c_hc, c_no_hc)

    def test_hc_refine_matches_manual_hc_on_gbest(self):
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        cfg = PSOConfig(hc_use_solver=True, hc_restarts=2, hc_max_iter_per_run=50, hc_use_two_opt=True)
        s = PSOSolver(dist, num_particles=6, seed=7, config=cfg)
        p, c, h = s.solve(max_iter=30)

        # run PSO without HC to obtain gbest, then run HillClimbing manually on the gbest
        cfg_no_hc = PSOConfig(hc_use_solver=False)
        s_no_hc = PSOSolver(dist, num_particles=6, seed=7, config=cfg_no_hc)
        gbest, gcost, gh = s_no_hc.solve(max_iter=30)

        # run HillClimbing directly from that gbest with same HC settings
        hc = HillClimbingSolver(dist, seed=999)
        hc_best_path, hc_best_cost, hc_history = hc.solve(
            max_iter=cfg.hc_max_iter_per_run,
            start_path=gbest,
            use_two_opt=cfg.hc_use_two_opt,
            restarts=cfg.hc_restarts,
            restart_strategy=cfg.hc_restart_strategy,
            restart_perturb_prob=cfg.hc_restart_perturb_prob,
            max_no_improve_per_run=cfg.hc_max_no_improve_per_run,
        )

        # The PSO result with HC should be at least as good as manual HC on the PSO gbest
        self.assertLessEqual(c, hc_best_cost)


if __name__ == '__main__':
    unittest.main()
