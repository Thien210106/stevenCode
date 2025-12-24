# HILL-CLIMBING (TSP) — PSO + Hill-Climbing Hybrid (Educational Project)

A small educational project demonstrating a Discrete PSO solver and a Hill-Climbing (Steepest Ascent) local search for the Traveling Salesman Problem (TSP). The repository includes a hybrid refinement where PSO can call the hill-climbing solver periodically or at the end to improve solutions.

---

## Quick start 🚀

- Run the demo CLI (example):

```bash
python main.py --n-cities 8 --seed 42 --pso-particles 30 --pso-iter 200 --periodic-hc-interval 10
```

- Run tests:

```bash
python -m unittest discover -s tests -v
```

---

## Recommended parameter settings ✅

These are conservative, general-purpose recommendations. Tweak them for your instance size and compute budget.

### PSO (in `PSOConfig` / CLI)

- `num_particles` (CLI `--pso-particles`): 20–50 — more particles increase diversity at cost of time. ⚖️
- `max_iter` (`--pso-iter`): 50–500 depending on problem size. 
- `p_personal_base`, `p_personal_scale`: Defaults (0.2, 0.3) — conservative personal learning.
- `p_global_base`, `p_global_scale`: Defaults (0.4, 0.4) — balance between personal/global influence.
- `perturb_prob`: 0.05–0.2 — small random swaps to escape local minima.
- `patience`: 10–50 — early stopping when gbest is stagnant.

### Hybrid Hill-Climbing (PSO -> HC)

- `hc_use_solver`: `True` to enable powerful hill-climbing refinement (recommended).
- `hc_restarts`: 1–5 — number of HC restarts used for refinement; higher yields better final solutions.
- `hc_max_iter_per_run`: 30–200 — local search budget per HC run.
- `hc_use_two_opt`: `True` to allow 2-opt improvements (recommended).
- `hc_interval`: 0 (disabled) or >0 to apply HC every N PSO iterations (try `5` or `10`).
- `hc_apply_to`: `'gbest'` or `'topk'` — use `'topk'` with `hc_top_k=2` to refine several good solutions.

### Hill Climbing (standalone)

- `restarts`: 2–10 for small problems; more for larger or harder instances.
- `use_two_opt`: `True` to help escape plateaus.
- `max_no_improve_per_run`: small integer (1–5) to short-circuit hopeless runs.

---

## Example outputs ✨

Running the demo with periodic HC often prints progress like:

```
[*] Running PSO...
Iteration 10: new gbest_cost = 145.0
Iteration 10: HC improved gbest to 138.0
Iteration 20: new gbest_cost = 130.0
...
--- PSO Result ---
Iterations: 42
Best path: [0, 3, 2, 1, 4, 5]
Best cost: 130.0
```

The hill-climbing result (standalone) looks like:

```
--- Hill Climbing Result ---
Iterations: 7
Best path: [3, 1, 4, 0, 2]
Best cost: 162
```

(Exact numbers depend on the RNG seed and instance.)

---

## Reproducibility & tips 🧪

- Use the `--seed` flag (or `PSOSolver(..., seed=...)`) to make runs repeatable.
- For larger instances (n &gt; 50) consider increasing `num_particles` and enabling `num_workers` in `PSOConfig` to parallelize cost evaluation.
- Use `hc_interval` carefully — frequent HC applications can increase runtime; tune `hc_max_iter_per_run` and `hc_restarts` accordingly.

---

## Where to go next

- Add benchmarks in `benchmarks/` to analyze runtime vs. quality across parameter grids.
- Add a small plotting script to visualize `history` outputs.
- Add a small CLI option to dump the full run history (JSON) for offline analysis.

---

## Programmatic usage & JSON output (example) 🧩

You can use the solver programmatically (recommended for experiments and automation) and serialize results as JSON for downstream analysis.

Example script (save as `dump_results.py`):

```python
from pso import PSOSolver, PSOConfig

# provide or generate a distance matrix (symmetric)
dist = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0],
]

cfg = PSOConfig(hc_use_solver=True, hc_use_two_opt=True)
solver = PSOSolver(dist, num_particles=10, seed=42, config=cfg)
best_path, best_cost, history = solver.solve(max_iter=50, patience=10)

# convert history to serializable structure
out = {
    "best_path": best_path,
    "best_cost": best_cost,
    "history": [{"path": p, "cost": c} for p, c in history],
}

import json
print(json.dumps(out))
```

Run and save to a file:

```bash
python dump_results.py > results.json
```

Example JSON output (trimmed):

```json
{
  "best_path": [0, 3, 2, 1],
  "best_cost": 42.0,
  "history": [
    {"path": [0, 1, 2, 3], "cost": 57.0},
    {"path": [0, 3, 2, 1], "cost": 42.0}
  ]
}
```

Notes & tips:
- For large problems, the `history` may be large; consider only saving the last N entries or summary statistics.
- Use `--seed` for reproducible runs and include the seed in the JSON if you plan multiple experiments.

## Algorithms implemented

- Hill Climbing (Steepest Ascent)
- Hill Climbing with Stochastic Restarts
- Discrete Particle Swarm Optimization (PSO)
- Hybrid / Memetic PSO (PSO + Hill Climbing refinement)
- Local search operators: swap, 2-opt

## Limitations

- Designed for educational purposes and small-to-medium TSP instances
- Discrete PSO is heuristic and does not guarantee optimality
- Runtime grows quickly with the number of cities due to local search
