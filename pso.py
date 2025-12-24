import random
import math
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable


@dataclass
class PSOConfig:
    """Configuration for PSO solver behavior."""
    p_personal_base: float = 0.2
    p_personal_scale: float = 0.3
    p_global_base: float = 0.4
    p_global_scale: float = 0.4
    perturb_prob: float = 0.1
    use_local_refine: bool = True
    local_refine_steps: int = 30
    move_operator: str = "swap"  # 'swap' or '2opt'
    two_opt_search_steps: int = 1  # number of two-opt improvements to attempt per call
    num_workers: int = 1  # placeholder for parallelism

    # HillClimbing integration options
    hc_use_solver: bool = True  # when True, use HillClimbingSolver for final refinement
    hc_restarts: int = 0
    hc_max_iter_per_run: int = 100
    hc_restart_strategy: str = "perturb"  # 'random' or 'perturb'
    hc_restart_perturb_prob: float = 0.2
    hc_use_two_opt: bool = False
    hc_max_no_improve_per_run: Optional[int] = None

    # Periodic HC during PSO
    hc_interval: int = 0  # every N iterations apply HC (0 = disabled)
    hc_apply_to: str = "gbest"  # 'gbest' or 'topk'
    hc_top_k: int = 1  # when hc_apply_to == 'topk', apply HC to top-k pbest particles


from hill_climbing import HillClimbingSolver


class PSOSolver:
    """
    Upgraded Discrete PSO solver for the Traveling Salesman Problem (TSP)

    Enhancements:
    - Configurable learning probabilities via PSOConfig
    - Random perturbation (escape local minima)
    - Hybrid PSO + local search refinement (configurable)
    - Pluggable move operators (swap, 2-opt)
    - Early stopping (patience)
    """

    def __init__(
        self,
        distance_matrix: List[List[float]],
        num_particles: int = 20,
        seed: Optional[int] = None,
        config: Optional[PSOConfig] = None,
    ):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        self.num_particles = num_particles
        self.rng = random.Random(seed)
        self.config = config or PSOConfig()

        # Validate initial inputs
        self._validate_matrix()

    # ================= CORE UTILITIES =================

    def calculate_cost(self, path: List[int]) -> float:
        total = 0.0
        for i in range(self.n - 1):
            total += self.matrix[path[i]][path[i + 1]]
        total += self.matrix[path[-1]][path[0]]
        return total

    def random_path(self) -> List[int]:
        return self.rng.sample(range(self.n), self.n)

    # ================= PSO OPERATORS =================

    def move_towards(self, current: List[int], target: List[int], prob: float) -> List[int]:
        """
        Move `current` closer to `target` using swap operations (O(n)).
        """
        new_path = current[:]
        index_map = {v: i for i, v in enumerate(new_path)}

        for i in range(self.n):
            if new_path[i] != target[i] and self.rng.random() < prob:
                j = index_map[target[i]]
                if j != i:
                    v_i = new_path[i]
                    new_path[i], new_path[j] = new_path[j], new_path[i]
                    index_map[v_i] = j
                    index_map[new_path[i]] = i

        return new_path

    def _validate_matrix(self) -> None:
        """Validate that distance_matrix is square, numeric, and non-negative."""
        if not isinstance(self.matrix, list):
            raise ValueError("distance_matrix must be a list of lists")
        if self.n < 2:
            raise ValueError("distance_matrix must contain at least 2 nodes")
        for row in self.matrix:
            if not isinstance(row, list) or len(row) != self.n:
                raise ValueError("distance_matrix must be square (n x n)")
            for val in row:
                if not isinstance(val, (int, float)):
                    raise ValueError("distance_matrix entries must be numeric")
                if val < 0:
                    raise ValueError("distance values must be non-negative")

    def random_perturb(self, path: List[int], prob: float = 0.1) -> List[int]:
        """
        Random swap with small probability to increase diversity.
        """
        if self.rng.random() < prob:
            i, j = self.rng.sample(range(self.n), 2)
            path[i], path[j] = path[j], path[i]
        return path

    # ================= PSO OPERATORS =================

    def two_opt_once(self, path: List[int]) -> List[int]:
        """Perform a single improving 2-opt move (first improvement strategy).

        Returns the improved path if found, otherwise returns the original path.
        """
        base_cost = self.calculate_cost(path)
        for i in range(0, self.n - 2):
            for j in range(i + 2, self.n):
                if j - i == 1:
                    continue
                candidate = path[:i] + path[i:j][::-1] + path[j:]
                cost = self.calculate_cost(candidate)
                if cost < base_cost:
                    return candidate
        return path

    # ================= LOCAL SEARCH =================

    def local_refine(self, path: List[int], steps: int = 30) -> Tuple[List[int], float]:
        """
        Simple Hill Climbing refinement applied to the global best (swap-based).
        """
        best = path[:]
        best_cost = self.calculate_cost(best)

        for _ in range(steps):
            i, j = self.rng.sample(range(self.n), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            cost = self.calculate_cost(candidate)

            if cost < best_cost:
                best, best_cost = candidate, cost

        return best, best_cost
    # ================= MAIN SOLVER =================

    def _compute_costs(self, paths: List[List[int]]) -> List[float]:
        """Compute costs for a list of paths; parallelized when config.num_workers>1."""
        if getattr(self, "config", None) and self.config.num_workers and self.config.num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.config.num_workers) as ex:
                return list(ex.map(self.calculate_cost, paths))
        else:
            return [self.calculate_cost(p) for p in paths]

    def solve(
        self,
        max_iter: int = 200,
        patience: Optional[int] = None,
        verbose: bool = False,
        callback: Optional[Callable[[int, List[int], float], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Tuple[List[int], float, List[Tuple[List[int], float]]]:

        # Initialize swarm
        particles = [self.random_path() for _ in range(self.num_particles)]
        pbest = [p[:] for p in particles]
        pbest_cost = self._compute_costs(particles)

        # Global best
        gbest_idx = pbest_cost.index(min(pbest_cost))
        gbest = pbest[gbest_idx][:]
        gbest_cost = pbest_cost[gbest_idx]

        history = [(gbest[:], gbest_cost)]

        no_improve = 0
        if patience is None:
            patience = math.inf

        # PSO loop
        for it in range(1, max_iter + 1):
            # check for external stop request
            if stop_event and stop_event.is_set():
                if verbose:
                    print(f"Stopping early at iteration {it} due to stop_event")
                break

            # Adaptive probabilities (configurable)
            alpha = 1 - it / max_iter
            prob_pbest = self.config.p_personal_base + self.config.p_personal_scale * alpha
            prob_gbest = self.config.p_global_base + self.config.p_global_scale * (1 - alpha)
            # clamp
            prob_pbest = max(0.0, min(1.0, prob_pbest))
            prob_gbest = max(0.0, min(1.0, prob_gbest))

            new_positions: List[List[int]] = []
            for i in range(self.num_particles):
                new_pos = self.move_towards(particles[i], pbest[i], prob_pbest)
                new_pos = self.move_towards(new_pos, gbest, prob_gbest)

                # optional 2-opt local move
                if self.config.move_operator == "2opt":
                    for _ in range(self.config.two_opt_search_steps):
                        new_pos = self.two_opt_once(new_pos)

                # random perturbation
                new_pos = self.random_perturb(new_pos, prob=self.config.perturb_prob)

                new_positions.append(new_pos)

            # compute costs in parallel when configured
            new_costs = self._compute_costs(new_positions)

            for i, new_pos in enumerate(new_positions):
                new_cost = new_costs[i]
                particles[i] = new_pos

                if new_cost < pbest_cost[i]:
                    pbest[i] = new_pos[:]
                    pbest_cost[i] = new_cost

            # Update global best
            current_best_cost = min(pbest_cost)
            if current_best_cost < gbest_cost:
                gbest_cost = current_best_cost
                gbest = pbest[pbest_cost.index(current_best_cost)][:]

                no_improve = 0
                if verbose:
                    print(f"Iteration {it}: new gbest_cost = {gbest_cost}")
            else:
                no_improve += 1

            # Optionally run periodic Hill Climbing refinement
            if self.config.hc_use_solver and self.config.hc_interval and (it % self.config.hc_interval == 0):
                if self.config.hc_apply_to not in ("gbest", "topk"):
                    raise ValueError("hc_apply_to must be 'gbest' or 'topk'")

                if self.config.hc_apply_to == "gbest":
                    hc_seed = self.rng.randrange(0, 2 ** 32)
                    hc = HillClimbingSolver(self.matrix, seed=hc_seed)
                    hc_best_path, hc_best_cost, hc_history = hc.solve(
                        max_iter=self.config.hc_max_iter_per_run,
                        start_path=gbest,
                        use_two_opt=self.config.hc_use_two_opt,
                        restarts=self.config.hc_restarts,
                        restart_strategy=self.config.hc_restart_strategy,
                        restart_perturb_prob=self.config.hc_restart_perturb_prob,
                        max_no_improve_per_run=self.config.hc_max_no_improve_per_run,
                        stop_event=stop_event,
                    )
                    if hc_best_cost < gbest_cost:
                        gbest, gbest_cost = hc_best_path, hc_best_cost
                        if verbose:
                            print(f"Iteration {it}: HC improved gbest to {gbest_cost}")
                    history.extend(hc_history)
                else:  # topk
                    # find indices of top-k pbest (smallest costs)
                    k = max(1, min(self.config.hc_top_k, len(pbest_cost)))
                    top_k_indices = sorted(range(len(pbest_cost)), key=lambda idx: pbest_cost[idx])[:k]
                    for idx in top_k_indices:
                        start = pbest[idx][:]
                        hc_seed = self.rng.randrange(0, 2 ** 32)
                        hc = HillClimbingSolver(self.matrix, seed=hc_seed)
                        hc_best_path, hc_best_cost, hc_history = hc.solve(
                            max_iter=self.config.hc_max_iter_per_run,
                            start_path=start,
                            use_two_opt=self.config.hc_use_two_opt,
                            restarts=self.config.hc_restarts,
                            restart_strategy=self.config.hc_restart_strategy,
                            restart_perturb_prob=self.config.hc_restart_perturb_prob,
                            max_no_improve_per_run=self.config.hc_max_no_improve_per_run,
                            stop_event=stop_event,
                        )
                        # update that particle's pbest if improved
                        if hc_best_cost < pbest_cost[idx]:
                            pbest[idx] = hc_best_path[:]
                            pbest_cost[idx] = hc_best_cost
                            if hc_best_cost < gbest_cost:
                                gbest, gbest_cost = hc_best_path[:], hc_best_cost
                                if verbose:
                                    print(f"Iteration {it}: HC on top-k idx {idx} improved gbest to {gbest_cost}")
                        history.extend(hc_history)

            history.append((gbest[:], gbest_cost))
            if callback:
                callback(it, gbest[:], gbest_cost)

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at iteration {it}")
                break
        # Final local refinement (Hybrid PSO + HC), controlled by config
        if self.config.use_local_refine:
            if self.config.hc_use_solver:
                # Use HillClimbingSolver for a potentially stronger refinement. Seed derived from RNG for reproducibility.
                hc_seed = self.rng.randrange(0, 2 ** 32)
                hc = HillClimbingSolver(self.matrix, seed=hc_seed)
                # call hill-climbing with restarts and configured options
                hc_best_path, hc_best_cost, hc_history = hc.solve(
                    max_iter=self.config.hc_max_iter_per_run,
                    start_path=gbest,
                    use_two_opt=self.config.hc_use_two_opt,
                    restarts=self.config.hc_restarts,
                    restart_strategy=self.config.hc_restart_strategy,
                    restart_perturb_prob=self.config.hc_restart_perturb_prob,
                    max_no_improve_per_run=self.config.hc_max_no_improve_per_run,
                    stop_event=stop_event,
                )
                gbest, gbest_cost = hc_best_path, hc_best_cost
                # append HC history to overall history for debugging/analysis
                history.extend(hc_history)
            else:
                gbest, gbest_cost = self.local_refine(gbest, steps=self.config.local_refine_steps)

        history.append((gbest[:], gbest_cost))
        return gbest, gbest_cost, history


# ================= TEST =================

if __name__ == "__main__":
    dist = [
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0],
    ]

    solver = PSOSolver(dist, num_particles=15, seed=42)
    best_path, best_cost, history = solver.solve(
        max_iter=200,
        patience=30,
        verbose=True
    )

    print("\nBest path:", best_path)
    print("Best cost:", best_cost)
