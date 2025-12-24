from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hill_climbing import HillClimbingSolver
from pso import PSOSolver, PSOConfig


# ----------------------------
# Problem generation utilities
# ----------------------------

def generate_euclidean_instance(n: int, seed: Optional[int]) -> Tuple[List[Tuple[float, float]], List[List[float]]]:
    """Generate random 2D points and an Euclidean distance matrix."""
    import random

    rng = random.Random(seed)
    pts = [(rng.random() * 100.0, rng.random() * 100.0) for _ in range(n)]
    mat: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = d
            mat[j][i] = d
    return pts, mat


def generate_random_symmetric_matrix(n: int, seed: Optional[int]) -> List[List[int]]:
    """Generate a symmetric random distance matrix (integers), mirroring main.py."""
    import random

    rng = random.Random(seed)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = rng.randint(10, 100)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


# ----------------------------
# GUI
# ----------------------------


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)

        self.distance_matrix: Optional[List[List[float]]] = None
        self.points: Optional[List[Tuple[float, float]]] = None  # when generated as Euclidean
        self._run_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._build_style()
        self._build_layout()
        self._set_defaults()

    # ---------- UI construction ----------

    def _build_style(self) -> None:
        style = ttk.Style(self.master)
        # use native theme when possible
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TLabel", padding=(2, 2))
        style.configure("TButton", padding=(6, 3))
        style.configure("TLabelframe", padding=(8, 6))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        # two columns: controls (left) and visuals (right)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.controls = ttk.Frame(self)
        self.controls.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        self.visuals = ttk.Frame(self)
        self.visuals.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.visuals.columnconfigure(0, weight=1)
        self.visuals.rowconfigure(0, weight=1)

        # Controls sections
        self._build_instance_box()
        self._build_algo_box()
        self._build_hc_box()
        self._build_pso_box()
        self._build_run_box()

        # Visuals: notebook
        self.nb = ttk.Notebook(self.visuals)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_matrix = ttk.Frame(self.nb)
        self.tab_history = ttk.Frame(self.nb)
        self.tab_tour = ttk.Frame(self.nb)
        self.tab_log = ttk.Frame(self.nb)

        self.nb.add(self.tab_matrix, text="Matrix")
        self.nb.add(self.tab_history, text="Cost History")
        self.nb.add(self.tab_tour, text="Tour")
        self.nb.add(self.tab_log, text="Log")

        self._build_matrix_tab()
        self._build_history_tab()
        self._build_tour_tab()
        self._build_log_tab()

    def _build_instance_box(self) -> None:
        lf = ttk.LabelFrame(self.controls, text="Problem Setup")
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Cities (n)").grid(row=0, column=0, sticky="w")
        self.var_n = tk.IntVar()
        self.spin_n = ttk.Spinbox(lf, from_=4, to=200, textvariable=self.var_n, width=8)
        self.spin_n.grid(row=0, column=1, sticky="w")

        ttk.Label(lf, text="Seed").grid(row=1, column=0, sticky="w")
        self.var_seed = tk.StringVar()
        self.ent_seed = ttk.Entry(lf, textvariable=self.var_seed, width=10)
        self.ent_seed.grid(row=1, column=1, sticky="w")

        ttk.Label(lf, text="Instance type").grid(row=2, column=0, sticky="w")
        self.var_inst_type = tk.StringVar()
        self.cmb_inst_type = ttk.Combobox(
            lf,
            textvariable=self.var_inst_type,
            values=["Euclidean points (plot tour)", "Random symmetric weights"],
            state="readonly",
            width=24,
        )
        self.cmb_inst_type.grid(row=2, column=1, sticky="w")

        btn_row = ttk.Frame(lf)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=1)

        ttk.Button(btn_row, text="Generate", command=self.on_generate).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btn_row, text="Load CSV", command=self.on_load_csv).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(btn_row, text="Save", command=self.on_save_instance).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        ttk.Label(lf, text="Tip: Euclidean enables tour plot.").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_algo_box(self) -> None:
        lf = ttk.LabelFrame(self.controls, text="Algorithm")
        lf.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Mode").grid(row=0, column=0, sticky="w")
        self.var_mode = tk.StringVar()
        self.cmb_mode = ttk.Combobox(
            lf,
            textvariable=self.var_mode,
            values=["Hill Climbing", "PSO", "Hybrid (PSO + Hill Climbing)"],
            state="readonly",
            width=24,
        )
        self.cmb_mode.grid(row=0, column=1, sticky="w")
        self.cmb_mode.bind("<<ComboboxSelected>>", lambda _e: self._sync_enabled_controls())

    def _build_hc_box(self) -> None:
        lf = ttk.LabelFrame(self.controls, text="Hill Climbing Params")
        lf.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        self.var_hc_max_iter = tk.IntVar()
        self.var_hc_restarts = tk.IntVar()
        self.var_hc_use_two_opt = tk.BooleanVar()
        self.var_hc_restart_strategy = tk.StringVar()
        self.var_hc_restart_perturb_prob = tk.DoubleVar()
        self.var_hc_max_no_improve = tk.StringVar()

        ttk.Label(lf, text="max_iter").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_hc_max_iter, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(lf, text="restarts").grid(row=1, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_hc_restarts, width=10).grid(row=1, column=1, sticky="w")

        ttk.Checkbutton(lf, text="use_two_opt", variable=self.var_hc_use_two_opt).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(lf, text="restart_strategy").grid(row=3, column=0, sticky="w")
        ttk.Combobox(lf, textvariable=self.var_hc_restart_strategy, values=["random", "perturb"], state="readonly", width=10).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(lf, text="perturb_prob").grid(row=4, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_hc_restart_perturb_prob, width=10).grid(row=4, column=1, sticky="w")

        ttk.Label(lf, text="max_no_improve").grid(row=5, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_hc_max_no_improve, width=10).grid(row=5, column=1, sticky="w")
        ttk.Label(lf, text="(blank = None)").grid(row=6, column=0, columnspan=2, sticky="w")

    def _build_pso_box(self) -> None:
        lf = ttk.LabelFrame(self.controls, text="PSO Params")
        lf.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        # basics
        self.var_pso_particles = tk.IntVar()
        self.var_pso_iter = tk.IntVar()
        self.var_pso_patience = tk.StringVar()

        ttk.Label(lf, text="particles").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_pso_particles, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(lf, text="max_iter").grid(row=1, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_pso_iter, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(lf, text="patience").grid(row=2, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_pso_patience, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(lf, text="(blank = inf)").grid(row=3, column=0, columnspan=2, sticky="w")

        # operator
        self.var_move_operator = tk.StringVar()
        self.var_two_opt_steps = tk.IntVar()
        self.var_perturb_prob = tk.DoubleVar()

        ttk.Label(lf, text="move_operator").grid(row=4, column=0, sticky="w")
        ttk.Combobox(lf, textvariable=self.var_move_operator, values=["swap", "2opt"], state="readonly", width=10).grid(
            row=4, column=1, sticky="w"
        )

        ttk.Label(lf, text="2opt_steps").grid(row=5, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_two_opt_steps, width=10).grid(row=5, column=1, sticky="w")

        ttk.Label(lf, text="perturb_prob").grid(row=6, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_perturb_prob, width=10).grid(row=6, column=1, sticky="w")

        # hybrid specifics
        ttk.Separator(lf).grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)

        self.var_use_local_refine = tk.BooleanVar()
        ttk.Checkbutton(lf, text="final_refine (local)", variable=self.var_use_local_refine).grid(row=8, column=0, columnspan=2, sticky="w")

        self.var_periodic_hc_interval = tk.IntVar()
        self.var_hc_apply_to = tk.StringVar()
        self.var_hc_top_k = tk.IntVar()

        ttk.Label(lf, text="periodic_hc_interval").grid(row=9, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_periodic_hc_interval, width=10).grid(row=9, column=1, sticky="w")

        ttk.Label(lf, text="hc_apply_to").grid(row=10, column=0, sticky="w")
        ttk.Combobox(lf, textvariable=self.var_hc_apply_to, values=["gbest", "topk"], state="readonly", width=10).grid(
            row=10, column=1, sticky="w"
        )

        ttk.Label(lf, text="hc_top_k").grid(row=11, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.var_hc_top_k, width=10).grid(row=11, column=1, sticky="w")

    def _build_run_box(self) -> None:
        lf = ttk.LabelFrame(self.controls, text="Run")
        lf.grid(row=4, column=0, sticky="ew")
        lf.columnconfigure(0, weight=1)

        self.btn_run = ttk.Button(lf, text="Run", command=self.on_run)
        self.btn_run.grid(row=0, column=0, sticky="ew")

        self.btn_stop = ttk.Button(lf, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.progress = ttk.Progressbar(lf, mode="determinate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(lf, textvariable=self.var_status, wraplength=260).grid(row=3, column=0, sticky="w", pady=(6, 0))

        ttk.Separator(lf).grid(row=4, column=0, sticky="ew", pady=8)

        self.var_best = tk.StringVar(value="Best: -")
        ttk.Label(lf, textvariable=self.var_best, wraplength=260).grid(row=5, column=0, sticky="w")

    def _build_matrix_tab(self) -> None:
        self.tab_matrix.columnconfigure(0, weight=1)
        self.tab_matrix.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self.tab_matrix, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(self.tab_matrix, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=yscroll.set)

        xscroll = ttk.Scrollbar(self.tab_matrix, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=xscroll.set)

    def _build_history_tab(self) -> None:
        self.tab_history.columnconfigure(0, weight=1)
        self.tab_history.rowconfigure(0, weight=1)

        self.fig_hist = Figure(figsize=(6, 4), dpi=100)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_hist.set_title("Best cost vs iteration")
        self.ax_hist.set_xlabel("Iteration")
        self.ax_hist.set_ylabel("Cost")

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.tab_history)
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_tour_tab(self) -> None:
        self.tab_tour.columnconfigure(0, weight=1)
        self.tab_tour.rowconfigure(0, weight=1)

        self.fig_tour = Figure(figsize=(6, 4), dpi=100)
        self.ax_tour = self.fig_tour.add_subplot(111)
        self.ax_tour.set_title("Tour (requires Euclidean instance)")
        self.ax_tour.set_xlabel("x")
        self.ax_tour.set_ylabel("y")

        self.canvas_tour = FigureCanvasTkAgg(self.fig_tour, master=self.tab_tour)
        self.canvas_tour.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_log_tab(self) -> None:
        self.tab_log.columnconfigure(0, weight=1)
        self.tab_log.rowconfigure(0, weight=1)

        self.txt_log = tk.Text(self.tab_log, wrap="word")
        self.txt_log.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(self.tab_log, orient="vertical", command=self.txt_log.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.txt_log.configure(yscrollcommand=yscroll.set)

        btn_row = ttk.Frame(self.tab_log)
        btn_row.grid(row=1, column=0, sticky="ew", pady=6)
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        ttk.Button(btn_row, text="Clear", command=lambda: self.txt_log.delete("1.0", "end")).grid(row=0, column=0, sticky="w")
        ttk.Button(btn_row, text="Export log", command=self.on_export_log).grid(row=0, column=1, sticky="e")

    def _set_defaults(self) -> None:
        self.var_n.set(12)
        self.var_seed.set("42")
        self.var_inst_type.set("Euclidean points (plot tour)")

        self.var_mode.set("Hybrid (PSO + Hill Climbing)")

        # HC
        self.var_hc_max_iter.set(200)
        self.var_hc_restarts.set(2)
        self.var_hc_use_two_opt.set(True)
        self.var_hc_restart_strategy.set("perturb")
        self.var_hc_restart_perturb_prob.set(0.2)
        self.var_hc_max_no_improve.set("")

        # PSO
        self.var_pso_particles.set(30)
        self.var_pso_iter.set(200)
        self.var_pso_patience.set("20")
        self.var_move_operator.set("swap")
        self.var_two_opt_steps.set(1)
        self.var_perturb_prob.set(0.1)
        self.var_use_local_refine.set(True)

        self.var_periodic_hc_interval.set(10)
        self.var_hc_apply_to.set("gbest")
        self.var_hc_top_k.set(1)

        self._sync_enabled_controls()
        self.on_generate()

    # ---------- Helpers ----------

    def _parse_seed(self) -> Optional[int]:
        s = (self.var_seed.get() or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            raise ValueError("Seed must be an integer (or blank)")

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")

    def _sync_enabled_controls(self) -> None:
        mode = self.var_mode.get()

        # enable/disable PSO section
        pso_state = "normal" if mode in ("PSO", "Hybrid (PSO + Hill Climbing)") else "disabled"
        hc_state = "normal" if mode in ("Hill Climbing", "Hybrid (PSO + Hill Climbing)") else "disabled"

        # brute-force: traverse widgets under frames
        def set_state_recursive(widget: tk.Widget, state: str) -> None:
            for child in widget.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
                set_state_recursive(child, state)

        # controls are in LabelFrames at fixed rows
        # (we keep references by searching text in children frames)
        # safer: apply by storing references to frames
        # Here, apply by manual: rows 2 and 3 are HC and PSO frames.
        hc_frame = self.controls.grid_slaves(row=2, column=0)[0]
        pso_frame = self.controls.grid_slaves(row=3, column=0)[0]

        set_state_recursive(hc_frame, hc_state)
        set_state_recursive(pso_frame, pso_state)

        # some widgets should always be enabled (labels etc.)
        # ttk.Label doesn't accept state; ignore.

    def _render_matrix(self) -> None:
        mat = self.distance_matrix
        self.tree.delete(*self.tree.get_children())
        if not mat:
            self.tree["columns"] = ()
            return

        n = len(mat)
        cols = [f"c{i}" for i in range(n)]
        self.tree["columns"] = cols
        for i, col in enumerate(cols):
            self.tree.heading(col, text=str(i))
            self.tree.column(col, width=48, anchor="center", stretch=False)

        for r in range(n):
            row = [f"{mat[r][c]:.2f}" if isinstance(mat[r][c], float) else str(mat[r][c]) for c in range(n)]
            self.tree.insert("", "end", values=row)

    def _render_history(self, costs: List[float]) -> None:
        self.ax_hist.clear()
        self.ax_hist.set_title("Best cost vs iteration")
        self.ax_hist.set_xlabel("Iteration")
        self.ax_hist.set_ylabel("Cost")
        if costs:
            self.ax_hist.plot(list(range(len(costs))), costs)
        self.canvas_hist.draw_idle()

    def _render_tour(self, path: Optional[List[int]]) -> None:
        self.ax_tour.clear()
        self.ax_tour.set_title("Tour (requires Euclidean instance)")
        self.ax_tour.set_xlabel("x")
        self.ax_tour.set_ylabel("y")

        if not self.points or not path:
            self.ax_tour.text(0.5, 0.5, "No coordinates available\n(generate Euclidean instance)", ha="center", va="center", transform=self.ax_tour.transAxes)
            self.canvas_tour.draw_idle()
            return

        pts = self.points
        xs = [pts[i][0] for i in path] + [pts[path[0]][0]]
        ys = [pts[i][1] for i in path] + [pts[path[0]][1]]

        self.ax_tour.plot(xs, ys, marker="o")
        for idx, (x, y) in enumerate(pts):
            self.ax_tour.annotate(str(idx), (x, y), textcoords="offset points", xytext=(6, 6))

        self.canvas_tour.draw_idle()

    def _read_matrix_from_csv(self, p: Path) -> List[List[float]]:
        with p.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        mat: List[List[float]] = []
        for r in rows:
            if not r:
                continue
            mat.append([float(x) for x in r])
        # basic validation
        n = len(mat)
        if n < 2 or any(len(row) != n for row in mat):
            raise ValueError("CSV matrix must be square (n x n)")
        return mat

    # ---------- Actions ----------

    def on_generate(self) -> None:
        try:
            n = int(self.var_n.get())
            if n < 4:
                raise ValueError("n must be >= 4")
            seed = self._parse_seed()

            if self.var_inst_type.get().startswith("Euclidean"):
                pts, mat = generate_euclidean_instance(n, seed)
                self.points = pts
                self.distance_matrix = mat
                self._log(f"Generated Euclidean instance: n={n}, seed={seed}")
            else:
                mat = generate_random_symmetric_matrix(n, seed)
                self.points = None
                self.distance_matrix = [[float(x) for x in row] for row in mat]
                self._log(f"Generated random symmetric matrix: n={n}, seed={seed}")

            self._render_matrix()
            self._render_history([])
            self._render_tour(None)
            self.var_best.set("Best: -")
            self.var_status.set("Ready")
        except Exception as e:
            messagebox.showerror("Generate failed", str(e))

    def on_load_csv(self) -> None:
        fp = filedialog.askopenfilename(
            title="Load distance matrix (CSV)",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            mat = self._read_matrix_from_csv(Path(fp))
            self.distance_matrix = mat
            self.points = None
            self.var_n.set(len(mat))
            self._render_matrix()
            self._render_history([])
            self._render_tour(None)
            self.var_best.set("Best: -")
            self._log(f"Loaded CSV matrix: {fp}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def on_save_instance(self) -> None:
        if not self.distance_matrix:
            messagebox.showwarning("Nothing to save", "Generate or load an instance first.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save instance",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
        )
        if not fp:
            return

        try:
            p = Path(fp)
            if p.suffix.lower() == ".csv":
                with p.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    for row in self.distance_matrix:
                        w.writerow(row)
            else:
                payload: dict[str, Any] = {
                    "distance_matrix": self.distance_matrix,
                    "points": self.points,
                }
                p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._log(f"Saved instance: {fp}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_export_log(self) -> None:
        fp = filedialog.asksaveasfilename(
            title="Export log",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            Path(fp).write_text(self.txt_log.get("1.0", "end"), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def on_stop(self) -> None:
        if self._run_thread and self._run_thread.is_alive():
            self._stop_flag.set()
            self._log("Stop requested. Will stop after current iteration (best-effort).")
            self.var_status.set("Stopping…")

    def on_run(self) -> None:
        if not self.distance_matrix:
            messagebox.showwarning("No instance", "Generate or load an instance first.")
            return
        if self._run_thread and self._run_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return

        try:
            # capture snapshot of params now
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        self._stop_flag.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.configure(value=0)
        self.var_status.set("Running…")
        self.var_best.set("Best: -")

        self._log("========================================")
        self._log(f"Run mode: {params['mode']}")
        self._log(f"n = {len(self.distance_matrix)}")

        self._run_thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self._run_thread.start()

    # ---------- Run logic ----------

    def _collect_params(self) -> dict[str, Any]:
        mode = self.var_mode.get()

        # hill climbing
        hc_max_iter = int(self.var_hc_max_iter.get())
        hc_restarts = int(self.var_hc_restarts.get())
        hc_use_two_opt = bool(self.var_hc_use_two_opt.get())
        hc_restart_strategy = self.var_hc_restart_strategy.get()
        hc_restart_perturb_prob = float(self.var_hc_restart_perturb_prob.get())

        max_no_improve_str = (self.var_hc_max_no_improve.get() or "").strip()
        hc_max_no_improve = int(max_no_improve_str) if max_no_improve_str else None

        # pso
        pso_particles = max(3, int(self.var_pso_particles.get()))
        pso_iter = int(self.var_pso_iter.get())
        patience_str = (self.var_pso_patience.get() or "").strip()
        pso_patience = int(patience_str) if patience_str else None

        move_operator = self.var_move_operator.get()
        two_opt_steps = int(self.var_two_opt_steps.get())
        perturb_prob = float(self.var_perturb_prob.get())

        use_local_refine = bool(self.var_use_local_refine.get())

        periodic_hc_interval = int(self.var_periodic_hc_interval.get())
        hc_apply_to = self.var_hc_apply_to.get()
        hc_top_k = int(self.var_hc_top_k.get())

        seed = self._parse_seed()

        if hc_restart_strategy not in ("random", "perturb"):
            raise ValueError("restart_strategy must be random or perturb")
        if move_operator not in ("swap", "2opt"):
            raise ValueError("move_operator must be swap or 2opt")
        if not (0.0 <= perturb_prob <= 1.0):
            raise ValueError("perturb_prob must be in [0, 1]")
        if periodic_hc_interval < 0:
            raise ValueError("periodic_hc_interval must be >= 0")

        return {
            "mode": mode,
            "seed": seed,
            "hc": {
                "max_iter": hc_max_iter,
                "restarts": hc_restarts,
                "use_two_opt": hc_use_two_opt,
                "restart_strategy": hc_restart_strategy,
                "restart_perturb_prob": hc_restart_perturb_prob,
                "max_no_improve": hc_max_no_improve,
            },
            "pso": {
                "particles": pso_particles,
                "max_iter": pso_iter,
                "patience": pso_patience,
                "move_operator": move_operator,
                "two_opt_steps": two_opt_steps,
                "perturb_prob": perturb_prob,
                "use_local_refine": use_local_refine,
                "periodic_hc_interval": periodic_hc_interval,
                "hc_apply_to": hc_apply_to,
                "hc_top_k": hc_top_k,
            },
        }

    def _run_worker(self, params: dict[str, Any]) -> None:
        try:
            mat = self.distance_matrix
            assert mat is not None

            mode = params["mode"]
            seed = params["seed"]
            hc_p = params["hc"]
            pso_p = params["pso"]

            best_path: List[int] = []
            best_cost: float = float("inf")
            history_costs: List[float] = []

            def ui(fn, *a, **kw):
                self.master.after(0, lambda: fn(*a, **kw))

            # update helpers
            def set_progress(cur: int, total: int):
                if total <= 0:
                    return
                self.progress.configure(maximum=total, value=min(cur, total))

            def set_best_text(path: List[int], cost: float):
                self.var_best.set(f"Best cost = {cost:.4f} | path = {path}")

            def finish_ok(status: str = "Done"):
                self.btn_run.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                self.var_status.set(status)

            def finish_fail(_msg: str):
                self.btn_run.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                self.var_status.set("Failed")

            # -------------- Hill Climbing --------------
            if mode == "Hill Climbing":
                ui(self._log, "Starting Hill Climbing…")
                total = hc_p["max_iter"] * (1 + max(0, int(hc_p["restarts"])))
                ui(set_progress, 0, total)

                best_so_far = float("inf")

                def hc_callback(iter_idx: int, path: List[int], cost: float):
                    nonlocal best_so_far
                    if self._stop_flag.is_set():
                        return
                    best_so_far = min(best_so_far, cost)
                    history_costs.append(best_so_far)

                    # Throttle UI updates
                    if iter_idx == 1 or iter_idx == total or (iter_idx % 5 == 0):
                        ui(set_progress, iter_idx, total)
                        ui(set_best_text, path, best_so_far)

                solver = HillClimbingSolver(mat, seed=seed)
                best_path, best_cost, history = solver.solve(
                    max_iter=hc_p["max_iter"],
                    restarts=hc_p["restarts"],
                    use_two_opt=hc_p["use_two_opt"],
                    restart_strategy=hc_p["restart_strategy"],
                    restart_perturb_prob=hc_p["restart_perturb_prob"],
                    max_no_improve_per_run=hc_p["max_no_improve"],
                    stop_event=self._stop_flag,
                    callback=hc_callback,
                )

                # Fallback if callback did not populate history_costs
                if not history_costs:
                    best_so_far = float("inf")
                    for _p, c in history:
                        best_so_far = min(best_so_far, c)
                        history_costs.append(best_so_far)

            # -------------- PSO / Hybrid --------------
            elif mode in ("PSO", "Hybrid (PSO + Hill Climbing)"):
                ui(self._log, "Starting PSO…" if mode == "PSO" else "Starting Hybrid PSO + Hill Climbing…")
                total = pso_p["max_iter"]
                ui(set_progress, 0, total)

                cfg = PSOConfig(
                    move_operator=pso_p["move_operator"],
                    two_opt_search_steps=pso_p["two_opt_steps"],
                    perturb_prob=pso_p["perturb_prob"],
                    use_local_refine=pso_p["use_local_refine"],

                    # Hybrid HC (enabled only in Hybrid mode)
                    hc_use_solver=(mode == "Hybrid (PSO + Hill Climbing)"),
                    hc_restarts=hc_p["restarts"],
                    hc_max_iter_per_run=hc_p["max_iter"],
                    hc_restart_strategy=hc_p["restart_strategy"],
                    hc_restart_perturb_prob=hc_p["restart_perturb_prob"],
                    hc_use_two_opt=hc_p["use_two_opt"],
                    hc_max_no_improve_per_run=hc_p["max_no_improve"],

                    # Periodic HC during PSO
                    hc_interval=pso_p["periodic_hc_interval"],
                    hc_apply_to=pso_p["hc_apply_to"],
                    hc_top_k=pso_p["hc_top_k"],
                )

                costs: List[float] = []

                def callback(iter_idx: int, gbest_path: List[int], gbest_cost: float):
                    if self._stop_flag.is_set():
                        return
                    costs.append(gbest_cost)

                    # Throttle UI updates
                    if iter_idx == 1 or iter_idx == total or (iter_idx % 5 == 0):
                        ui(set_progress, iter_idx, total)
                        ui(set_best_text, gbest_path, gbest_cost)

                solver = PSOSolver(mat, num_particles=pso_p["particles"], seed=seed, config=cfg)
                best_path, best_cost, history = solver.solve(
                    max_iter=pso_p["max_iter"],
                    patience=pso_p["patience"],
                    verbose=False,
                    callback=callback,
                    stop_event=self._stop_flag,
                )

                # Prefer per-iteration callback series; fall back to solver history if needed.
                if costs:
                    history_costs = costs
                else:
                    best_so_far = float("inf")
                    for _p, c in history:
                        best_so_far = min(best_so_far, c)
                        history_costs.append(best_so_far)

                ui(self._log, f"PSOConfig = {asdict(cfg)}")

            else:
                ui(finish_fail, f"Unknown mode: {mode}")
                ui(messagebox.showerror, "Run failed", f"Unknown mode: {mode}")
                return

            # ----------- Render results -----------
            ui(self._render_history, history_costs)
            ui(self._render_tour, best_path)
            ui(set_best_text, best_path, best_cost)

            if self._stop_flag.is_set():
                ui(self._log, "Stopped by user.")
                ui(finish_ok, "Stopped")
            else:
                ui(finish_ok, "Done")
                ui(self._log, f"Finished. Best cost = {best_cost:.4f}")

        except Exception as e:
            ui(finish_fail, str(e))
            ui(self._log, f"Error: {e}")
            ui(messagebox.showerror, "Run failed", str(e))

def main() -> None:
    root = tk.Tk()
    root.title("TSP — Hill Climbing / PSO (GUI)")
    root.geometry("1180x720")

    # Improve scaling on Windows high-DPI
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
