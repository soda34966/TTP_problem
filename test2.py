import numpy as np
import warnings
import os
import copy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# --- Optimization Libraries ---
import optuna 
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.indicators.hv import HV

warnings.filterwarnings("ignore")

# ==========================================
# 1. TSP SOLVER (Iterated Local Search)
# ==========================================
def fast_2opt(tour, dist_matrix):
    """
    Standard 2-opt local search to uncross edges.
    """
    tour = np.array(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if j == n and i == 0: continue
                d_curr = dist_matrix[tour[i], tour[i+1]] + dist_matrix[tour[j%n], tour[(j+1)%n]]
                d_new = dist_matrix[tour[i], tour[j%n]] + dist_matrix[tour[i+1], tour[(j+1)%n]]
                if d_new < d_curr - 1e-6:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    return tour

def double_bridge_kick(tour):
    """
    Perturbation mechanism to escape local optima.
    """
    n = len(tour)
    if n < 8: return tour
    pos = sorted(np.random.choice(range(1, n-1), 3, replace=False))
    p1, p2, p3 = pos
    return np.concatenate((tour[:p1], tour[p3:], tour[p2:p3], tour[p1:p2]))

def get_perfect_tour(n_cities, dist_matrix):
    """
    Solves the TSP component using ILS (Iterated Local Search).
    """
    print("  > Optimizing Base Tour (Target: < 2600)...")
    global_best_tour = None
    global_best_len = float('inf')
    
    # 5 Restarts to ensure we don't get stuck in a bad valley
    for attempt in range(5):
        # Greedy Initialization
        curr = np.random.randint(0, n_cities)
        tour = [curr]
        unvisited = np.ones(n_cities, dtype=bool)
        unvisited[curr] = False
        for _ in range(n_cities - 1):
            dists = np.where(unvisited, dist_matrix[curr], np.inf)
            nearest = np.argmin(dists)
            tour.append(nearest)
            unvisited[nearest] = False
            curr = nearest
        
        current_tour = fast_2opt(tour, dist_matrix)
        current_len = np.sum(dist_matrix[current_tour, np.roll(current_tour, -1)])
        
        # ILS Kicks
        no_improv = 0
        while no_improv < 15:
            cand_tour = double_bridge_kick(current_tour)
            cand_tour = fast_2opt(cand_tour, dist_matrix)
            cand_len = np.sum(dist_matrix[cand_tour, np.roll(cand_tour, -1)])
            
            if cand_len < current_len:
                current_len = cand_len
                current_tour = cand_tour
                no_improv = 0
            else:
                no_improv += 1
        
        if current_len < global_best_len:
            global_best_len = current_len
            global_best_tour = current_tour
            print(f"    [Restart {attempt+1}] Found new best: {global_best_len}")
            if global_best_len < 2600: break 
            
    return global_best_tour

# ==========================================
# 2. TTP PROBLEM (Dual Heuristics)
# ==========================================
class TTPProblem(Problem):
    def __init__(self, filename, perfect_tour=None, dist_exponent=1.0, weight_penalty=1.0):
        self.filename = filename
        self._load_data(filename)
        
        if perfect_tour is None: return

        self.perfect_tour = perfect_tour
        self.dist_to_end = self._calc_dist_to_end(self.perfect_tour)
        
        # --- DUAL HEURISTIC SCORING ---
        self.scores_efficiency = np.zeros(self.n_items) # For Left Side (Fast)
        self.scores_knapsack = np.zeros(self.n_items)   # For Right Side (Rich)
        
        epsilon = 1e-5
        
        for i in range(self.n_items):
            p, w, node = self.item_profits[i], self.item_weights[i], self.item_locs[i]
            dist = self.dist_to_end[node]
            
            # 1. Efficiency Score: Penalize Weight & Distance (Strict)
            w_term = np.power(w, weight_penalty) if w > 0 else epsilon
            d_term = np.power(dist + 1.0, dist_exponent)
            self.scores_efficiency[i] = p / (w_term * d_term)

            # 2. Knapsack Score: Only Profit/Weight (Loose)
            self.scores_knapsack[i] = p / (w if w > 0 else epsilon)
            
        # Normalize both
        self._norm(self.scores_efficiency)
        self._norm(self.scores_knapsack)

        super().__init__(n_var=self.n_cities + self.n_items, n_obj=2, xl=0, xu=1)

    def _norm(self, arr):
        if arr.max() > arr.min():
            arr[:] = (arr - arr.min()) / (arr.max() - arr.min())

    def _load_data(self, filename):
        with open(filename, 'r') as f: lines = [l.strip() for l in f.readlines()]
        self.nodes = []
        self.items = []
        node_sec = False
        for line in lines:
            if line.startswith("CAPACITY"): self.capacity = float(line.split(":")[1])
            if line.startswith("MIN SPEED"): self.v_min = float(line.split(":")[1])
            if line.startswith("MAX SPEED"): self.v_max = float(line.split(":")[1])
            if line.startswith("NODE_COORD_SECTION"): node_sec = True; continue
            if line.startswith("ITEMS SECTION"): node_sec = False; continue
            parts = line.split()
            if not parts or not parts[0].isdigit(): continue
            if node_sec: self.nodes.append([float(parts[1]), float(parts[2])])
            else: self.items.append([float(parts[1]), float(parts[2]), int(parts[3])-1])
        self.n_cities = len(self.nodes)
        self.n_items = len(self.items)
        self.items = np.array(self.items)
        self.dist_matrix = np.ceil(cdist(self.nodes, self.nodes)) 
        self.item_locs = self.items[:, 2].astype(int)
        self.item_weights = self.items[:, 1]
        self.item_profits = self.items[:, 0]

    def _calc_dist_to_end(self, tour):
        dists = np.zeros(self.n_cities)
        cum = 0
        for i in range(len(tour)-1, 0, -1):
            u, v = tour[i-1], tour[i]
            cum += self.dist_matrix[u, v]
            dists[u] = cum
        dists[tour[-1]] = self.dist_matrix[tour[-1], tour[0]]
        return dists

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = len(X)
        F = np.zeros((n_pop, 2))
        cap = self.capacity
        v_diff = self.v_max - self.v_min
        
        for i in range(n_pop):
            tour = np.argsort(X[i, :self.n_cities])
            pack_genes = X[i, self.n_cities:]
            picked_mask = pack_genes > 0.5
            
            # --- GREEDY REPAIR ---
            # Drop items based on Knapsack Score if over capacity
            current_w = np.sum(self.item_weights[picked_mask])
            if current_w > cap:
                idxs = np.where(picked_mask)[0]
                sorted_idxs = idxs[np.argsort(self.scores_knapsack[idxs])] 
                for idx in sorted_idxs:
                    w = self.item_weights[idx]
                    current_w -= w
                    picked_mask[idx] = False
                    if current_w <= cap: break

            current_p = np.sum(self.item_profits[picked_mask])
            
            tour_rolled = np.roll(tour, -1)
            dists = self.dist_matrix[tour, tour_rolled]
            city_weights = np.zeros(self.n_cities)
            if np.any(picked_mask):
                np.add.at(city_weights, self.item_locs[picked_mask], self.item_weights[picked_mask])
            
            cur_loads = np.cumsum(city_weights[tour])
            velocities = self.v_max - (cur_loads / cap) * v_diff
            velocities = np.clip(velocities, self.v_min, self.v_max)
            time = np.sum(dists / velocities)
            F[i, 0] = time
            F[i, 1] = -current_p
        out["F"] = F

# ==========================================
# 3. HYBRID SAMPLING (ANCHORED)
# ==========================================
class HybridSampling(Sampling):
    def __init__(self, problem):
        super().__init__()
        self.perfect_tour = problem.perfect_tour
        self.scores_eff = problem.scores_efficiency
        self.scores_knap = problem.scores_knapsack
        
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        n_c = problem.n_cities
        
        perfect_keys = np.zeros(n_c)
        ranks = np.argsort(self.perfect_tour)
        for i in range(n_c): perfect_keys[i] = ranks[i] / n_c
            
        for i in range(n_samples):
            # Tour Protection
            X[i, :n_c] = perfect_keys + np.random.normal(0, 0.00001, n_c)
            
            # --- ANCHORED PACKING ---
            if i == 0:
                X[i, n_c:] = 0.0 # Anchor Left (Empty)
                continue
            if i == n_samples - 1:
                X[i, n_c:] = 1.0 # Anchor Right (Full -> Repair will fix)
                continue

            # Linear Density between anchors
            density = i / (n_samples - 1)
            
            # Score Mixing: < 60% use Efficiency, > 60% use Knapsack
            ratio = i / n_samples
            if ratio < 0.6:
                chosen_scores = self.scores_eff
            else:
                chosen_scores = self.scores_knap
            
            threshold = 1.0 - density
            genes = chosen_scores - threshold + 0.5 + np.random.normal(0, 0.05, problem.n_items)
            X[i, n_c:] = np.clip(genes, 0, 1)
            
        return X

class HybridMutation(Mutation):
    def __init__(self, problem, mut_prob=0.05):
        super().__init__()
        self.scores_eff = problem.scores_efficiency
        self.scores_knap = problem.scores_knapsack
        self.prob = mut_prob
        
    def _do(self, problem, X, **kwargs):
        n_c = problem.n_cities
        for i in range(len(X)):
            if np.random.random() < 0.05: 
                idx1, idx2 = np.random.randint(0, n_c, 2)
                X[i, idx1], X[i, idx2] = X[i, idx2], X[i, idx1]

            # Determine Mutation Strategy based on individual "fullness"
            fullness = np.mean(X[i, n_c:])
            
            if fullness < 0.5:
                use_scores = self.scores_eff
            else:
                use_scores = self.scores_knap
                
            mask = np.random.random(problem.n_items) < self.prob
            idxs = np.where(mask)[0]
            for idx in idxs:
                score = use_scores[idx]
                curr_val = X[i, n_c + idx]
                if score > 0.7: X[i, n_c + idx] = min(1.0, curr_val + 0.4)
                elif score < 0.3: X[i, n_c + idx] = max(0.0, curr_val - 0.4)
                else: X[i, n_c + idx] = 1.0 - curr_val
        return X

# ==========================================
# 4. VISUALIZATION
# ==========================================
def visualize_trajectory(problem, solution_x, title="Balanced Solution Route"):
    """
    Visualizes the physical route and item pickups for a specific solution.
    """
    # 1. Decode Tour
    n_cities = problem.n_cities
    tour_keys = solution_x[:n_cities]
    tour = np.argsort(tour_keys)
    
    # 2. Decode Packing (Re-run the Greedy Repair to know exactly what was kept)
    pack_genes = solution_x[n_cities:]
    picked_mask = pack_genes > 0.5
    
    # --- REPEAT REPAIR LOGIC ---
    current_w = np.sum(problem.item_weights[picked_mask])
    if current_w > problem.capacity:
        idxs = np.where(picked_mask)[0]
        # Sort by Knapsack Score (Low to High), drop lowest first
        sorted_idxs = idxs[np.argsort(problem.scores_knapsack[idxs])] 
        for idx in sorted_idxs:
            w = problem.item_weights[idx]
            current_w -= w
            picked_mask[idx] = False
            if current_w <= problem.capacity: break
    # ---------------------------

    # 3. Setup Plot
    coords = np.array(problem.nodes)
    plt.figure(figsize=(12, 10))
    
    # 4. Plot The Route (Black Line)
    route_tour = np.append(tour, tour[0]) 
    plt.plot(coords[route_tour, 0], coords[route_tour, 1], 
             c='k', alpha=0.3, linewidth=1, zorder=1, label='Route')

    # 5. Plot Cities (Small grey dots)
    plt.scatter(coords[:, 0], coords[:, 1], c='gray', s=10, alpha=0.5, zorder=2)

    # 6. Plot Picked Items (Green Circles)
    picked_locs = problem.item_locs[picked_mask]
    
    total_profit = np.sum(problem.item_profits[picked_mask])
    total_weight = np.sum(problem.item_weights[picked_mask])
    
    if len(picked_locs) > 0:
        plt.scatter(coords[picked_locs, 0], coords[picked_locs, 1], 
                    c='green', s=50, edgecolors='black', zorder=3, label='Item Picked')

    # 7. Highlight Start/End (Red Star)
    start_node = tour[0]
    plt.scatter(coords[start_node, 0], coords[start_node, 1], 
                c='red', s=200, marker='*', zorder=4, label='Start/End')

    # Annotation
    info_text = (f"Total Profit: {total_profit:.0f}\n"
                 f"Final Weight: {total_weight:.1f}/{problem.capacity:.1f}\n"
                 f"Items Picked: {np.sum(picked_mask)}")
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(title)
    plt.legend(loc='lower right')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.savefig("TTP_Route_Map.png", dpi=150)
    print("\n> Route Map saved as 'TTP_Route_Map.png'")

# ==========================================
# 5. OPTIMIZATION LOOP
# ==========================================
def objective(trial, problem_path, ref_tour):
    # Tuning Ranges
    alpha = trial.suggest_float("dist_exponent", 1.0, 3.0) 
    w_pen = trial.suggest_float("weight_penalty", 1.0, 1.4) 
    mut_prob = trial.suggest_float("mut_prob", 0.02, 0.08)
    
    problem = TTPProblem(problem_path, ref_tour, dist_exponent=alpha, weight_penalty=w_pen)
    
    algorithm = SMSEMOA(
        pop_size=50, 
        sampling=HybridSampling(problem),
        crossover=SBX(eta=15, prob=0.9),
        mutation=HybridMutation(problem, mut_prob=mut_prob),
        eliminate_duplicates=DefaultDuplicateElimination()
    )
    
    res = minimize(problem, algorithm, ('n_gen', 60), verbose=False)
    F = res.F
    if F is None or len(F) == 0: return 0.0
    
    # Reference point to calculate Hypervolume
    ind = HV(ref_point=np.array([7000, 0]))
    return ind(F)

def main():
    # --- CONFIGURATION ---
    # NOTE: Update this path to where your TTP file is located
    problem_path = "TTP_resoure/a280-n279.txt"
    hpi_file = "HPI_a280-n279.f"
    
    if not os.path.exists(problem_path):
        print(f"Error: File '{problem_path}' not found.")
        print("Please place the TTP data file in the correct directory.")
        return

    print(">>> Generating Superior Base Tour...")
    temp_prob = TTPProblem(problem_path, perfect_tour=None) 
    ref_tour = get_perfect_tour(temp_prob.n_cities, temp_prob.dist_matrix)
    
    print(f"\n{'='*50}\n STARTING BAYESIAN OPTIMIZATION \n{'='*50}")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, problem_path, ref_tour), n_trials=10) 
    best = study.best_params
    print(f"\n>>> Best Params: {best}")
    
    print(f"\n{'='*50}\n STARTING FINAL RUN (Anchored Hybrid) \n{'='*50}")
    
    final_problem = TTPProblem(problem_path, ref_tour, 
                               dist_exponent=best["dist_exponent"],
                               weight_penalty=best["weight_penalty"])
    
    final_algo = SMSEMOA(
        pop_size=200, 
        sampling=HybridSampling(final_problem),
        crossover=SBX(eta=20, prob=0.9),
        mutation=HybridMutation(final_problem, mut_prob=best["mut_prob"]),
        eliminate_duplicates=DefaultDuplicateElimination()
    )
    
    res = minimize(final_problem, final_algo, ('n_gen', 600), seed=42, verbose=True)
    
    # --- POST-PROCESSING & PLOTTING ---
    sorted_indices = np.argsort(res.F[:, 0])
    F_sorted = res.F[sorted_indices]
    X_sorted = res.X[sorted_indices]

    # 1. Plot Pareto Front
    plt.figure(figsize=(10, 6))
    if os.path.exists(hpi_file):
        try:
            hpi = np.loadtxt(hpi_file)
            if hpi[0, 1] > 0: hpi[:, 1] = -hpi[:, 1]
            hpi = hpi[np.argsort(hpi[:, 0])]
            plt.plot(hpi[:, 0], hpi[:, 1], 'g--', linewidth=2, label='HPI Reference')
        except: pass

    plt.plot(F_sorted[:, 0], F_sorted[:, 1], 'r-o', linewidth=2, markersize=4, label='Anchored Solution')
    
    # Highlight the chosen "Balanced" solution
    mid_idx = len(F_sorted) // 2
    plt.scatter(F_sorted[mid_idx, 0], F_sorted[mid_idx, 1], c='blue', s=100, zorder=10, label='Selected for Map')

    plt.xlabel("Travel Time"); plt.ylabel("Negative Profit")
    plt.title(f"TTP Result: Anchored Hybrid Heuristics")
    plt.legend(); plt.grid(True, alpha=0.5)
    plt.savefig("TTP_Pareto_Front.png")
    print("\n> Pareto Front saved as 'TTP_Pareto_Front.png'")

    # 2. Visualize the Balanced Route
    print(f"\n>>> Visualizing Solution #{mid_idx} (Balanced Trade-off)...")
    visualize_trajectory(final_problem, X_sorted[mid_idx])

if __name__ == "__main__":
    main()