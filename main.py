import argparse
import random
from typing import List

from hill_climbing import HillClimbingSolver
from pso import PSOSolver, PSOConfig


def generate_dummy_data(n_cities: int, seed: int = None) -> List[List[int]]:
    """Generate a symmetric random distance matrix (integers)."""
    rng = random.Random(seed)
    matrix = [[0] * n_cities for _ in range(n_cities)]
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = rng.randint(10, 100)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def print_matrix(mat: List[List[int]]) -> None:
    for row in mat:
        print(" ".join(f"{v:3d}" for v in row))


def main():
    parser = argparse.ArgumentParser(description="Demo: TSP with Hill Climbing and PSO")
    parser.add_argument("--n-cities", type=int, default=6, help="number of cities")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--pso-particles", type=int, default=20, help="number of PSO particles")
    parser.add_argument("--pso-iter", type=int, default=100, help="PSO max iterations")

    # PSOConfig parameters
    parser.add_argument("--p-personal-base", type=float, default=0.2, help="personal influence base probability")
    parser.add_argument("--p-personal-scale", type=float, default=0.3, help="personal influence scale")
    parser.add_argument("--p-global-base", type=float, default=0.4, help="global influence base probability")
    parser.add_argument("--p-global-scale", type=float, default=0.4, help="global influence scale")
    parser.add_argument("--perturb-prob", type=float, default=0.1, help="random perturbation probability per particle")
    parser.add_argument("--no-local-refine", action="store_false", dest="use_local_refine", help="disable final local_refine step")
    parser.add_argument("--local-refine-steps", type=int, default=30, help="steps for local_refine when enabled")
    parser.add_argument("--move-operator", choices=["swap", "2opt"], default="swap", help="move operator used during PSO updates")
    parser.add_argument("--two-opt-search-steps", type=int, default=1, help="number of 2-opt attempts per particle update when enabled")
    parser.add_argument("--num-workers", type=int, default=1, help="parallel workers for cost evaluation (ThreadPoolExecutor)")

    # HC / hybrid options
    parser.add_argument("--no-hc", action="store_true", help="disable HillClimbing solver for hybrid refinement")
    parser.add_argument("--hc-restarts", type=int, default=1, help="HC restarts for hybrid refinement")
    parser.add_argument("--hc-max-iter-per-run", type=int, default=50, help="max iter per HC run when used by hybrid refinement")
    parser.add_argument("--hc-restart-strategy", choices=["random", "perturb"], default="perturb", help="restart strategy for HC")
    parser.add_argument("--hc-restart-perturb-prob", type=float, default=0.2, help="probability to perturb HC start when using 'perturb' strategy")
    parser.add_argument("--hc-use-two-opt", action="store_true", help="enable 2-opt in HC when used by hybrid refinement")
    parser.add_argument("--hc-max-no-improve-per-run", type=int, default=None, help="early stop threshold for HC runs (None = disabled)")
    parser.add_argument("--periodic-hc-interval", type=int, default=0, help="apply HC every N PSO iterations (0 = disabled)")
    parser.add_argument("--hc-apply-to", choices=["gbest", "topk"], default="gbest", help="apply periodic HC to gbest or top-k pbest")
    parser.add_argument("--hc-top-k", type=int, default=1, help="when hc-apply-to is topk, number of top particles to refine")

    # Output options
    parser.add_argument("--dump-json", type=str, default=None, help="Path to write JSON results (optional)")

    args = parser.parse_args()

    print("==========================================")
    print("  TSP DEMO — Hill Climbing + PSO")
    print("==========================================")

    # Generate reproducible data
    print(f"[*] Generating random distance matrix for {args.n_cities} cities (seed={args.seed})...")
    dist_matrix = generate_dummy_data(args.n_cities, seed=args.seed)
    print("Distance matrix:")
    print_matrix(dist_matrix)

    # Run hill-climbing example
    print("\n[*] Running Hill Climbing...")
    hc = HillClimbingSolver(dist_matrix, seed=args.seed)
    hc_best_path, hc_best_cost, hc_history = hc.solve(max_iter=200, restarts=args.hc_restarts, use_two_opt=True)

    print("\n--- Hill Climbing Result ---")
    print(f"Iterations: {len(hc_history) - 1}")
    print(f"Best path: {hc_best_path}")
    print(f"Best cost: {hc_best_cost}")

    # Build PSOConfig from CLI args
    print("\n[*] Running PSO...")
    cfg = PSOConfig(
        p_personal_base=args.p_personal_base,
        p_personal_scale=args.p_personal_scale,
        p_global_base=args.p_global_base,
        p_global_scale=args.p_global_scale,
        perturb_prob=args.perturb_prob,
        use_local_refine=args.use_local_refine,
        local_refine_steps=args.local_refine_steps,
        move_operator=args.move_operator,
        two_opt_search_steps=args.two_opt_search_steps,
        num_workers=args.num_workers,
        hc_use_solver=not args.no_hc,
        hc_restarts=args.hc_restarts,
        hc_max_iter_per_run=args.hc_max_iter_per_run,
        hc_restart_strategy=args.hc_restart_strategy,
        hc_restart_perturb_prob=args.hc_restart_perturb_prob,
        hc_use_two_opt=args.hc_use_two_opt,
        hc_max_no_improve_per_run=args.hc_max_no_improve_per_run,
        hc_interval=args.periodic_hc_interval,
        hc_apply_to=args.hc_apply_to,
        hc_top_k=args.hc_top_k,
    )

    pso = PSOSolver(dist_matrix, num_particles=max(3, args.pso_particles), seed=args.seed, config=cfg)
    pso_best_path, pso_best_cost, pso_history = pso.solve(max_iter=args.pso_iter, patience=20, verbose=True)

    # Optionally dump JSON
    if args.dump_json:
        import json
        out = {
            "best_path": pso_best_path,
            "best_cost": pso_best_cost,
            "history": [{"path": p, "cost": c} for p, c in pso_history],
        }
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Results written to {args.dump_json}")

    print("\n--- PSO Result ---")
    print(f"Iterations: {len(pso_history) - 1}")
    print(f"Best path: {pso_best_path}")
    print(f"Best cost: {pso_best_cost}")

    print("\nSample of final history entries (last 5):")
    for idx, (p, c) in enumerate(pso_history[-5:], start=max(0, len(pso_history) - 5)):
        print(f"{idx:4d}: cost={c:6.2f} path={p}")

    print("\n==========================================")


if __name__ == "__main__":
    main()