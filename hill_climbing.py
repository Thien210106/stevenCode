import random
import threading
from typing import List, Tuple, Optional, Callable


class HillClimbingSolver:
    """Hill Climbing (Steepest Ascent) solver for TSP.

    Upgrades to fit with PSO module:
    - seedable RNG via `seed`
    - input validation
    - optional `start_path` to resume/refine from a provided solution
    - small 2-opt helper for local improvements
    """

    def __init__(self, distance_matrix: List[List[float]], seed: Optional[int] = None):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        self.rng = random.Random(seed)
        self._validate_matrix()

    def _validate_matrix(self) -> None:
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

    def calculate_cost(self, path: List[int]) -> float:
        """Compute total length of closed tour `path`."""
        total = 0.0
        for i in range(self.n - 1):
            total += self.matrix[path[i]][path[i + 1]]
        total += self.matrix[path[-1]][path[0]]
        return total

    def get_neighbors(self, current_path: List[int]) -> List[List[int]]:
        """Generate all swap neighbors of `current_path`."""
        neighbors = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                neighbor = current_path[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def two_opt_once(self, path: List[int]) -> List[int]:
        """Perform a single first-improvement 2-opt move; return path if improved."""
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

    def _single_run(
        self,
        max_iter: int,
        start_path: List[int],
        use_two_opt: bool,
        max_no_improve: Optional[int],
        start_iter: int = 0,
        stop_event: Optional[threading.Event] = None,
        callback: Optional[Callable[[int, List[int], float], None]] = None,
    ) -> Tuple[List[int], float, List[Tuple[List[int], float]], int]:
        """Run a single hill-climb starting from `start_path`.

        Returns (best_path, best_cost, history_for_run, iterations_performed).
        """
        current_path = start_path[:]
        current_cost = self.calculate_cost(current_path)
        history: List[Tuple[List[int], float]] = [(current_path[:], current_cost)]

        no_improve = 0
        if max_no_improve is None:
            max_no_improve = float("inf")

        iterations = 0
        for it in range(max_iter):
            # support cooperative stop
            if stop_event and stop_event.is_set():
                break

            iterations += 1
            global_it = start_iter + iterations

            if use_two_opt:
                current_path = self.two_opt_once(current_path)
                current_cost = self.calculate_cost(current_path)

            neighbors = self.get_neighbors(current_path)

            best_neighbor = current_path
            best_neighbor_cost = current_cost
            found_better = False

            for neighbor in neighbors:
                cost = self.calculate_cost(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor_cost = cost
                    best_neighbor = neighbor
                    found_better = True

            if not found_better:
                no_improve += 1
                if callback:
                    callback(global_it, current_path[:], current_cost)
                if no_improve >= max_no_improve:
                    break
                else:
                    # still allow further iterations (could optionally randomize)
                    continue

            # reset no_improve
            no_improve = 0
            current_path = best_neighbor
            current_cost = best_neighbor_cost
            history.append((current_path[:], current_cost))

            if callback:
                callback(global_it, current_path[:], current_cost)

        return current_path, current_cost, history, iterations

    def solve(
        self,
        max_iter: int = 1000,
        start_path: Optional[List[int]] = None,
        use_two_opt: bool = False,
        restarts: int = 0,
        restart_strategy: str = "random",  # 'random' or 'perturb'
        restart_perturb_prob: float = 0.2,
        max_no_improve_per_run: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        callback: Optional[Callable[[int, List[int], float], None]] = None,
    ) -> Tuple[List[int], float, List[Tuple[List[int], float]]]:
        """Run steepest-ascent hill climbing with optional stochastic restarts.

        Args:
            max_iter: maximum iterations per run
            start_path: optional starting permutation for the first run
            use_two_opt: apply one 2-opt pass per iteration before evaluating neighbors
            restarts: number of additional random/perturbed restarts to perform (>=0)
            restart_strategy: 'random' to sample new random starts, 'perturb' to perturb best so far
            restart_perturb_prob: when using 'perturb', probability of swapping a pair to create a new start
            max_no_improve_per_run: early-stop threshold for no improvements per run
            stop_event: optional threading.Event for cooperative stop
            callback: optional callback called as callback(iter_idx, path, cost)

        Returns:
            (best_path, best_cost, history)
        """
        # Validate restart strategy
        if restart_strategy not in ("random", "perturb"):
            raise ValueError("restart_strategy must be 'random' or 'perturb'")

        overall_best_path = None
        overall_best_cost = float("inf")
        overall_history: List[Tuple[List[int], float]] = []

        # Prepare initial start paths for each run
        total_runs = 1 + max(0, int(restarts))

        iter_counter = 0
        for run in range(total_runs):
            # allow cooperative stop between runs
            if stop_event and stop_event.is_set():
                break

            # Determine start_path for this run
            if run == 0 and start_path is not None:
                sp = start_path[:]
            elif run == 0:
                sp = self.rng.sample(range(self.n), self.n)
            else:
                if restart_strategy == "random":
                    sp = self.rng.sample(range(self.n), self.n)
                else:  # perturb
                    if overall_best_path is None:
                        sp = self.rng.sample(range(self.n), self.n)
                    else:
                        sp = overall_best_path[:]
                        if self.rng.random() < restart_perturb_prob:
                            i, j = self.rng.sample(range(self.n), 2)
                            sp[i], sp[j] = sp[j], sp[i]

            best_path, best_cost, history, iters = self._single_run(
                max_iter,
                sp,
                use_two_opt,
                max_no_improve_per_run,
                start_iter=iter_counter,
                stop_event=stop_event,
                callback=callback,
            )

            iter_counter += iters

            # append run history
            overall_history.extend(history)

            # update overall best
            if best_cost < overall_best_cost:
                overall_best_cost = best_cost
                overall_best_path = best_path[:]

        # consistency check
        if overall_best_path is None:
            # fallback: return random path
            overall_best_path = self.rng.sample(range(self.n), self.n)
            overall_best_cost = self.calculate_cost(overall_best_path)

        return overall_best_path, overall_best_cost, overall_history

