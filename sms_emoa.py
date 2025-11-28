import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import concurrent.futures
import multiprocessing
import time
import glob

# --- Optimization Libraries ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling 
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.operators.crossover.sbx import SBX

warnings.filterwarnings("ignore")

# ==========================================
# 1. ROBUST VECTORIZED MATH
# ==========================================
def get_tour_length(tour, nodes):
    coords = nodes[tour]
    diffs = coords - np.roll(coords, -1, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def full_deterministic_2opt(tour, nodes):
    """
    O(N^2) Deterministic Scan.
    Checks EVERY possible swap. Guaranteed to find local optimum.
    Use this for N < 2000.
    """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 2):
            p1 = coords[i]
            p2 = coords[i+1]
            
            # Full window lookahead
            j_indices = np.arange(i + 2, n)
            if len(j_indices) == 0: continue
            
            p3s = coords[j_indices]
            # Wrap around handling for p4
            p4_indices = (j_indices + 1) % n
            p4s = coords[p4_indices]
            
            # Vectorized Distance Calc
            d_curr = np.sqrt(np.sum((p1 - p2)**2)) + np.sqrt(np.sum((p3s - p4s)**2, axis=1))
            d_swap = np.sqrt(np.sum((p1 - p3s)**2, axis=1)) + np.sqrt(np.sum((p2 - p4s)**2, axis=1))
            
            diffs = d_swap - d_curr
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < -1e-6:
                j = j_indices[min_idx]
                best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                coords = nodes[best_tour] # Update coords immediately
                improved = True
                # Restart on first improvement (First Choice Hill Climbing)
                break 
    return best_tour

def randomized_vectorized_2opt(tour, nodes, window=1000, max_scans=3000):
    """
    Approximation Scan.
    Checks random indices. Fast for massive N > 2000.
    """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    n_scans = min(max_scans, n)
    start_indices = np.random.randint(0, n - 3, size=n_scans)
    improved = False
    
    for i in start_indices:
        p1 = coords[i]; p2 = coords[i+1]
        j_start = i + 2; j_end = min(i + window, n - 1)
        if j_start >= j_end: continue
        
        j_indices = np.arange(j_start, j_end)
        p3s = coords[j_indices]; p4s = coords[j_indices + 1]
        
        d_curr = np.sqrt(np.sum((p1 - p2)**2)) + np.sqrt(np.sum((p3s - p4s)**2, axis=1))
        d_swap = np.sqrt(np.sum((p1 - p3s)**2, axis=1)) + np.sqrt(np.sum((p2 - p4s)**2, axis=1))
        
        diffs = d_swap - d_curr
        min_idx = np.argmin(diffs)
        
        if diffs[min_idx] < -1e-6:
            j = j_indices[min_idx]
            best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
            coords = nodes[best_tour]
            improved = True
    return best_tour

def double_bridge_kick(tour):
    n = len(tour)
    if n < 8: return tour
    pos = np.sort(np.random.choice(range(1, n-1), 4, replace=False))
    return np.concatenate((tour[:pos[0]], tour[pos[2]:pos[3]], tour[pos[1]:pos[2]], tour[pos[0]:pos[1]], tour[pos[3]:]))

# ==========================================
# 2. ADAPTIVE ILS WORKER
# ==========================================
def ils_worker(args):
    nodes, seed, time_limit = args
    np.random.seed(seed)
    n = len(nodes)
    start_time = time.time()
    
    # Init NN Tour
    start_node = np.random.randint(0, n)
    tree = cKDTree(nodes)
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)
    curr = start_node; tour[0] = curr; visited[curr] = True
    k_search = 100 if n > 10000 else n
    
    for i in range(1, n):
        dists, indices = tree.query(nodes[curr], k=k_search)
        if np.isscalar(indices): indices = [indices]
        found = False
        for idx in indices:
            if not visited[idx]:
                tour[i] = idx; visited[idx] = True; curr = idx; found = True; break
        if not found:
            rem = np.where(~visited)[0]
            if len(rem) > 0:
                idx = rem[0]; tour[i] = idx; visited[idx] = True; curr = idx

    best_tour = tour.copy()
    best_len = get_tour_length(best_tour, nodes)
    
    # ADAPTIVE STRATEGY
    use_full_scan = (n < 2000) # Threshold for switching to Deterministic
    
    while (time.time() - start_time) < time_limit:
        if use_full_scan:
            best_tour = full_deterministic_2opt(best_tour, nodes)
        else:
            best_tour = randomized_vectorized_2opt(best_tour, nodes, window=2000)
            
        curr_len = get_tour_length(best_tour, nodes)
        if curr_len < best_len: best_len = curr_len
        
        kicked = double_bridge_kick(best_tour)
        if use_full_scan:
            repaired = full_deterministic_2opt(kicked, nodes)
        else:
            repaired = randomized_vectorized_2opt(kicked, nodes, window=1500)
            
        d = get_tour_length(repaired, nodes)
        if d < best_len:
            best_len = d; best_tour = repaired
            
    return best_tour, best_len

def run_parallel_ils(nodes, time_limit=30):
    print(f"   > Starting Parallel ILS ({len(nodes)} cities) | Limit: {time_limit}s...")
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    tasks = [(nodes, np.random.randint(0, 10000) + i, time_limit) for i in range(n_workers)]
    best_global_tour = None; best_global_len = float('inf')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(ils_worker, tasks)
        for tour, score in results:
            print(f"     Worker Finished | Score: {score:.1f}")
            if score < best_global_len: best_global_len = score; best_global_tour = tour
    print(f"   > Best Tour Found: {best_global_len:.1f}")
    return best_global_tour

# ==========================================
# 3. TUNABLE SPECTRUM SAMPLING
# ==========================================
class TunableSpectrumSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        n_c = problem.n_cities
        
        base_tour = problem.base_tour
        perfect_keys = np.linspace(0.001, 0.999, n_c)
        def tour_to_keys(t):
            k = np.zeros(n_c); k[t] = perfect_keys; return k
        base_k = tour_to_keys(base_tour)

        coords = problem.nodes[base_tour]
        edges = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        dist_map = np.zeros(n_c); cum_dist = 0
        for i in range(n_c - 1, -1, -1):
            dist_map[base_tour[i]] = cum_dist; cum_dist += edges[i]
        item_dists = dist_map[problem.item_locs]

        for i in range(n_samples):
            # Tour: Gaussian Cloud
            noise = np.random.normal(0, 0.003, n_c)
            X[i, :n_c] = np.clip(base_k + noise, 0, 1)
            
            # Spectrum Packing
            target_pct = i / (n_samples - 1)
            target_weight = problem.capacity * target_pct
            
            # Tunable Alpha: 1.0 (Distance Matters) -> 0.0 (Greedy)
            alpha = 1.0 - target_pct
            tuned_cost = problem.item_weights * (item_dists ** alpha + 1e-9)
            tuned_scores = problem.item_profits / (tuned_cost + 1e-9)
            
            ranked_items = np.argsort(-tuned_scores)
            current_w = 0
            for idx in ranked_items:
                w = problem.item_weights[idx]
                if current_w + w <= target_weight:
                    X[i, n_c + idx] = 1.0; current_w += w
        return X

# ==========================================
# 4. OPERATORS
# ==========================================
class ProxyProblem:
    def __init__(self, xl, xu):
        self.xl = xl; self.xu = xu; self.n_var = len(xl)

class SmartMutation(Mutation):
    def __init__(self, problem):
        super().__init__()
        self.n_cities = problem.n_cities
        self.pack_prob = 0.15
        self.tour_prob = 0.15 # Increased tour mutation slightly
        self.scores = problem.physics_scores
        self.profit_scores = problem.item_profits

    def _do(self, problem, X, **kwargs):
        n_pop, _ = X.shape
        for i in range(n_pop):
            # Packing
            if np.random.random() < self.pack_prob:
                pack_genes = X[i, self.n_cities:]
                picked = pack_genes > 0.5
                use_physics = np.random.random() < 0.3 # 30% Physics, 70% Profit
                scores = self.scores if use_physics else self.profit_scores
                
                if np.any(picked):
                    indices = np.where(picked)[0]
                    worst = indices[np.argmin(scores[indices])]
                    X[i, self.n_cities + worst] = 0.0
                
                not_picked = ~picked
                if np.any(not_picked):
                    indices = np.where(not_picked)[0]
                    candidates = indices[np.argsort(scores[indices])[-10:]]
                    best = np.random.choice(candidates)
                    X[i, self.n_cities + best] = 1.0

            # Tour
            if np.random.random() < self.tour_prob:
                a, b = np.random.randint(0, self.n_cities, 2)
                if a > b: a, b = b, a
                X[i, a:b] = X[i, a:b][::-1]
        return X

class SmoothCrossover(Crossover):
    def __init__(self, problem):
        super().__init__(2, 2)
        self.n_cities = problem.n_cities
        self.sbx = SBX(prob=0.9, eta=15)

    def _do(self, problem, X, **kwargs):
        _, _, n_var = X.shape
        X_tour = X[:, :, :self.n_cities]
        X_pack = X[:, :, self.n_cities:]
        
        proxy = ProxyProblem(problem.xl[:self.n_cities], problem.xu[:self.n_cities])
        Y_tour = self.sbx._do(proxy, X_tour, **kwargs)
        
        mask = np.random.random(X_pack[0].shape) < 0.5
        Y_pack = np.zeros_like(X_pack)
        Y_pack[0] = np.where(mask, X_pack[0], X_pack[1])
        Y_pack[1] = np.where(mask, X_pack[1], X_pack[0])
        
        Y = np.zeros_like(X)
        Y[:, :, :self.n_cities] = Y_tour
        Y[:, :, self.n_cities:] = Y_pack
        return Y

# ==========================================
# 5. PROBLEM & MAIN
# ==========================================
class TTPProblem(Problem):
    def __init__(self, filename, base_tour):
        self.filename = filename
        self.base_tour = base_tour 
        self._load_data(filename)
        self._precalc_physics() 
        super().__init__(n_var=self.n_cities + self.n_items, n_obj=2, xl=0, xu=1)

    def _load_data(self, filename):
        with open(filename, 'r') as f: lines = [l.strip() for l in f.readlines()]
        self.nodes = []; self.items = []
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
        self.nodes = np.array(self.nodes); self.items = np.array(self.items)
        self.n_cities = len(self.nodes); self.n_items = len(self.items)
        self.item_locs = self.items[:, 2].astype(int)
        self.item_weights = self.items[:, 1]; self.item_profits = self.items[:, 0]

    def _precalc_physics(self):
        tour = self.base_tour
        coords = self.nodes[tour]
        edges = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        dist_map = np.zeros(self.n_cities); cum_dist = 0
        for i in range(self.n_cities - 1, -1, -1):
            dist_map[tour[i]] = cum_dist; cum_dist += edges[i]
        dists = dist_map[self.item_locs]
        cost = self.item_weights * dists
        if cost.max() > cost.min(): cost = (cost - cost.min()) / (cost.max() - cost.min())
        self.physics_scores = self.item_profits / (cost + 1e-6)
        self.ratio_scores = self.item_profits / (self.item_weights + 1e-9)

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = len(X); F = np.zeros((n_pop, 2)); cap = self.capacity; v_diff = self.v_max - self.v_min
        item_w = self.item_weights; item_p = self.item_profits
        item_l = self.item_locs; r_scores = self.ratio_scores; nodes = self.nodes
        
        for i in range(n_pop):
            tour = np.argsort(X[i, :self.n_cities])
            pack_genes = X[i, self.n_cities:]
            picked_mask = pack_genes > 0.5
            
            # Greedy Repair (Profit/Weight)
            current_w = np.sum(item_w[picked_mask])
            if current_w > cap:
                idxs = np.where(picked_mask)[0]
                sorted_idxs = idxs[np.argsort(r_scores[idxs])]
                ws = item_w[sorted_idxs]; cum_drop = np.cumsum(ws)
                excess = current_w - cap; drop_count = np.searchsorted(cum_drop, excess) + 1
                picked_mask[sorted_idxs[:drop_count]] = False
            
            current_p = np.sum(item_p[picked_mask])
            city_weights = np.zeros(self.n_cities)
            if np.any(picked_mask): np.add.at(city_weights, item_l[picked_mask], item_w[picked_mask])
            coords_ordered = nodes[tour]; weights_ordered = city_weights[tour]
            dists = np.sqrt(np.sum((coords_ordered - np.roll(coords_ordered, -1, axis=0))**2, axis=1))
            cur_loads = np.cumsum(weights_ordered)
            velocities = np.clip(self.v_max - (cur_loads / cap) * v_diff, self.v_min, self.v_max)
            F[i, 0] = np.sum(dists / velocities); F[i, 1] = -current_p
        out["F"] = F

def main():
    resource_dir = "TTP_resoure"; hpi_dir = "HPI"
    res_f_dir = "result_f"; res_graph_dir = "result_graph"
    os.makedirs(res_f_dir, exist_ok=True); os.makedirs(res_graph_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(resource_dir, "*.txt")))
    if not files: print("No files found."); return
    print(f"\n{'='*40}\n Available TTP Problems \n{'='*40}")
    for i, f in enumerate(files): print(f" [{i}] {os.path.basename(f)}")
    try:
        idx = int(input("Select problem index: "))
        selected_path = files[idx]
        file_name = os.path.basename(selected_path)
        base_name = file_name.replace(".txt", "")
    except: return

    temp_problem = type('obj', (object,), {'nodes': []})
    with open(selected_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        start = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"): start = True; continue
            if line.startswith("ITEMS SECTION"): start = False; continue
            if start and line[0].isdigit():
                parts = line.split()
                temp_problem.nodes.append([float(parts[1]), float(parts[2])])
    n_cities = len(temp_problem.nodes)
    temp_problem.nodes = np.array(temp_problem.nodes)
    
    ils_time = max(30, min(600, int(n_cities / 20))) 
    best_tour = run_parallel_ils(temp_problem.nodes, time_limit=ils_time)
    
    problem = TTPProblem(selected_path, best_tour)
    
    print(">>> Sampling Initial Population (Tunable Spectrum)...")
    sampler = TunableSpectrumSampling()
    X_init = sampler._do(problem, 200)
    out_init = {}; problem._evaluate(X_init, out_init); F_init = out_init["F"]

    print(f">>> Initializing SMS-EMOA...")
    algorithm = SMSEMOA(
        pop_size=100,
        sampling=X_init,
        crossover=SmoothCrossover(problem),
        mutation=SmartMutation(problem),
        eliminate_duplicates=NoDuplicateElimination()
    )

    print(f">>> Starting Optimization...")
    res = minimize(problem, algorithm, ('n_gen', 500), seed=42, verbose=True)

    f_path = os.path.join(res_f_dir, f"RES_{base_name}.f")
    sorted_indices = np.argsort(res.F[:, 0])
    F_sorted = res.F[sorted_indices]
    
    print(f"\n>>> Saving Results to {f_path}...")
    with open(f_path, 'w') as f_out:
        for row in F_sorted: f_out.write(f"{row[0]:.2f} {-row[1]:.2f}\n")

    graph_path = os.path.join(res_graph_dir, f"PLOT_{base_name}.png")
    plt.figure(figsize=(12, 8))
    hpi_file = os.path.join(hpi_dir, f"HPI_{base_name}.f")
    if os.path.exists(hpi_file):
        try:
            hpi = np.loadtxt(hpi_file)
            if hpi.ndim > 1:
                if hpi[0, 1] > 0: hpi[:, 1] = -hpi[:, 1]
                hpi = hpi[np.argsort(hpi[:, 0])]
                plt.plot(hpi[:, 0], hpi[:, 1], 'g--', linewidth=2.5, alpha=0.9, label='HPI Reference')
        except: pass

    plt.scatter(F_init[:, 0], F_init[:, 1], c='gray', marker='x', s=40, alpha=0.4, label='Initial')
    plt.plot(F_sorted[:, 0], F_sorted[:, 1], 'r-o', linewidth=1.5, markersize=5, label='Optimized')
    plt.xlabel("Travel Time"); plt.ylabel("Negative Profit")
    plt.title(f"TTP Evolution: {base_name}")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(graph_path, dpi=150)
    print(">>> Done.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()