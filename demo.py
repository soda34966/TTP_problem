from TTPProblem import TTPProblemAdapter
import numpy as np
import warnings
import os
import glob
import multiprocessing
import matplotlib.pyplot as plt

from problem_logic import CustomCrossover, SmartMutation, TTPProblem, TunableSpectrumSampling
from sms_logic import optimize_tour_orientation, run_parallel_ils

# ----------------------------
# Utility / HV (2D)
# ----------------------------
def hv_2d(F_list, ref):
    arr = np.array(F_list)
    # keep only points that are <= ref on both objectives (minimization)
    arr = arr[(arr[:,0] <= ref[0]) & (arr[:,1] <= ref[1])]
    if arr.size == 0:
        return 0.0
    arr = arr[arr[:,0].argsort()]
    hv = 0.0
    prev_x = ref[0]
    for x, y in arr:
        dx = prev_x - x
        dy = ref[1] - y
        if dx > 0 and dy > 0:
            hv += dx * dy
        prev_x = x
    return hv

# ----------------------------
# Keys <-> Permutation helpers
# ----------------------------
def keys_to_perm(keys):
    # keys: array of length n_cities, returns permutation array (0..n-1)
    return np.argsort(keys)

def perm_to_keys(perm, n_cities):
    # map a permutation to "key" representation (same scheme as your sampler)
    perfect_keys = np.linspace(0.00001, 0.99999, n_cities)
    k = np.zeros(n_cities)
    k[perm] = perfect_keys
    return k

# ----------------------------
# Order Crossover (OX) on tours (using keys)
# ----------------------------
def ox_crossover_keys(parentA, parentB, n_cities):
    # parentA, parentB: 1D arrays containing keys in first n_cities
    pA = keys_to_perm(parentA[:n_cities])
    pB = keys_to_perm(parentB[:n_cities])
    n = n_cities
    a, b = sorted(np.random.choice(range(n), 2, replace=False))
    child = -np.ones(n, dtype=int)

    # copy segment a..b from pA
    child[a:b+1] = pA[a:b+1]

    # fill remaining positions from pB in order
    cur = 0
    for val in pB:
        if val not in child:
            while child[cur] != -1:
                cur += 1
            child[cur] = val

    return perm_to_keys(child, n_cities)

# ----------------------------
# 2-opt local improvement on permutation
# ----------------------------
def two_opt_on_perm(perm, nodes, max_iters=10):
    # perm: 1D array of city indices in visiting order
    n = len(perm)
    if n <= 3:
        return perm
    improved = True
    it = 0
    # Precompute coords for speed
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                a_idx = perm[i - 1]
                b_idx = perm[i]
                c_idx = perm[j - 1]
                d_idx = perm[j % n]
                a = nodes[a_idx]; b = nodes[b_idx]; c = nodes[c_idx]; d = nodes[d_idx]
                # current = |a-b| + |c-d|, new = |a-c| + |b-d|
                if np.linalg.norm(a - b) + np.linalg.norm(c - d) > np.linalg.norm(a - c) + np.linalg.norm(b - d):
                    perm[i:j] = perm[i:j][::-1]
                    improved = True
                    break
            if improved:
                break
    return perm

# ----------------------------
# Improved tournament selection
# ----------------------------
def tournament_advanced(pop, F, tour_size=3, ref_point=None):
    # pop: list of individuals (1D arrays)
    # F: list/array of objective vectors (minimization)
    n = len(pop)
    idxs = np.random.choice(n, min(tour_size, n), replace=False)
    f_cand = [F[i] for i in idxs]

    def dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    # find non-dominated candidates among the tournament
    nd = []
    for j, fj in enumerate(f_cand):
        dominated = False
        for k, fk in enumerate(f_cand):
            if k == j: continue
            if dominates(fk, fj):
                dominated = True
                break
        if not dominated:
            nd.append((idxs[j], fj))

    if len(nd) == 0:
        # fallback: return a random candidate
        return pop[np.random.choice(idxs)]
    if len(nd) == 1:
        return pop[nd[0][0]]

    # tie-breaker: hypervolume contribution inside the non-dominated subset
    if ref_point is None:
        # global ref fallback: use worst observed + margin
        ref_point = (np.max(F, axis=0) * 1.05) + 1e-6

    f_nd = [item[1] for item in nd]
    # compute hv contributions among nd
    hv_all = hv_2d(f_nd, ref_point)
    contribs = []
    for i in range(len(f_nd)):
        others = [f for j,f in enumerate(f_nd) if j != i]
        contribs.append(hv_all - hv_2d(others, ref_point))
    best_idx_local = np.argmax(contribs)
    best_global_idx = nd[best_idx_local][0]
    return pop[best_global_idx]

# ----------------------------
# Packing local improvement (small swap-based local search)
# ----------------------------
def packing_local_search(keys, n_cities, items_locs, item_weights, item_profits, capacity, max_tries=20):
    # keys: full solution vector (keys for tour then 0/1 for items)
    pack = keys[n_cities:].astype(float).copy()
    picked = pack > 0.5
    current_w = np.sum(item_weights[picked])
    current_profit = np.sum(item_profits[picked])

    # try simple swap: remove one picked low-ratio, add one not-picked high-ratio
    if current_w >= capacity:
        return keys  # nothing simple to do
    # precompute ratio
    ratios = item_profits / (item_weights + 1e-9)
    not_picked_idxs = np.where(~picked)[0]
    picked_idxs = np.where(picked)[0]

    # try to add good items until capacity or tries exhausted
    tries = 0
    for idx in np.argsort(-ratios[not_picked_idxs]):
        if tries >= max_tries: break
        real_idx = not_picked_idxs[idx]
        w = item_weights[real_idx]
        if current_w + w <= capacity:
            pack[real_idx] = 1.0
            current_w += w
            current_profit += item_profits[real_idx]
        tries += 1

    keys[n_cities:] = pack
    return keys

# ----------------------------
# Main improved SMS-EMOA
# ----------------------------
def smsemoa_improved(problem,
                     n_var,
                     n_obj,
                     pop_size=40,
                     n_gen=500,
                     sampling=None,
                     crossover=None,
                     mutation=None,
                     tour_mut_prob=0.3,
                     pack_mut_prob=0.25,
                     tour_2opt_iters=5,
                     tournament_size=3,
                     seed=None):
    """
    Improved SMS-EMOA:
      - Uses OX on tours (keys -> perm -> OX -> keys) + uniform packing crossover
      - 2-opt local search on tours as mutation
      - packing local search / mutation
      - tournament selection (size 3) with HV tie-breaker
      - fixed population size with HV-based deletion
    Accepts 'problem' which must implement:
      - evaluate(x) -> np.array([f1,f2])  (minimization)
      - attributes: n_cities, n_items, nodes, item_locs, item_weights, item_profits, capacity
    """
    if seed is not None:
        np.random.seed(seed)

    # helpers to access fields (works for both adapter and original problem)
    n_cities = getattr(problem, "n_cities")
    n_items = getattr(problem, "n_items", None)
    if n_items is None:
        # fallback compute from n_var
        n_items = n_var - n_cities

    nodes = getattr(problem, "nodes")
    item_locs = getattr(problem, "item_locs")
    item_weights = getattr(problem, "item_weights")
    item_profits = getattr(problem, "item_profits")
    capacity = getattr(problem, "capacity")

    # --- initialize population (with sampling if provided) ---
    if sampling is not None:
        X_init = sampling._do(problem, pop_size)
        pop = [X_init[i].copy() for i in range(pop_size)]
    else:
        pop = [np.random.rand(n_var) for _ in range(pop_size)]

    # evaluate initial population
    F = [problem.evaluate(ind) for ind in pop]

    # dynamic reference point (updated each generation)
    ref_point = np.max(F, axis=0) * 1.05 + 1e-6

    for gen in range(n_gen):
        # update ref_point from current population
        ref_point = np.max(F, axis=0) * 1.05 + 1e-9

        # select parents using improved tournament
        p1 = tournament_advanced(pop, F, tour_size=tournament_size, ref_point=ref_point)
        p2 = tournament_advanced(pop, F, tour_size=tournament_size, ref_point=ref_point)

        # crossover: if user provided operator, use it; else use OX+uniform packing
        if crossover is not None:
            parents_array = np.vstack([p1, p2])  # shape (2, n_var)
            children = crossover._do(problem, parents_array)
            child1, child2 = children[0].copy(), children[1].copy()
        else:
            # OX on tours
            child1 = p1.copy()
            child2 = p2.copy()
            child1[:n_cities] = ox_crossover_keys(p1, p2, n_cities)
            child2[:n_cities] = ox_crossover_keys(p2, p1, n_cities)
            # uniform crossover for packing part
            mask = np.random.rand(n_items) < 0.5
            child1[n_cities:] = np.where(mask, p1[n_cities:], p2[n_cities:])
            child2[n_cities:] = np.where(mask, p2[n_cities:], p1[n_cities:])

        # mutation: if provided use that, else apply our built-in plan
        if mutation is not None:
            offm = np.vstack([child1, child2])
            outm = mutation._do(problem, offm)
            child1, child2 = outm[0].copy(), outm[1].copy()
        else:
            # tour mutation: 2-opt with some probability
            if np.random.rand() < tour_mut_prob:
                perm1 = keys_to_perm(child1[:n_cities])
                perm1 = two_opt_on_perm(perm1, nodes, max_iters=tour_2opt_iters)
                child1[:n_cities] = perm_to_keys(perm1, n_cities)
            if np.random.rand() < tour_mut_prob:
                perm2 = keys_to_perm(child2[:n_cities])
                perm2 = two_opt_on_perm(perm2, nodes, max_iters=tour_2opt_iters)
                child2[:n_cities] = perm_to_keys(perm2, n_cities)

            # packing mutation: occasional random flips and local search
            if np.random.rand() < pack_mut_prob:
                # flip a few random bits for child1
                flip_idx = np.random.choice(n_items, max(1, n_items // 50), replace=False)
                child1[n_cities + flip_idx] = 1.0 - child1[n_cities + flip_idx]
                child1 = packing_local_search(child1, n_cities, item_locs, item_weights, item_profits, capacity)
            if np.random.rand() < pack_mut_prob:
                flip_idx = np.random.choice(n_items, max(1, n_items // 50), replace=False)
                child2[n_cities + flip_idx] = 1.0 - child2[n_cities + flip_idx]
                child2 = packing_local_search(child2, n_cities, item_locs, item_weights, item_profits, capacity)

        # Evaluate offspring
        offspring = [child1, child2]
        F_off = [problem.evaluate(c) for c in offspring]

        # Merge and then remove (len(pop)+2 -> len(pop)) using HV contribution deletions
        pop += offspring
        F += F_off

        # Remove exactly two worst contributors to return to fixed size
        # But it's okay to remove one-by-one twice
        for _ in range(2):
            contributions = []
            hv_all = hv_2d(F, ref_point)
            for i in range(len(pop)):
                temp_F = [f for j, f in enumerate(F) if j != i]
                hv_without = hv_2d(temp_F, ref_point)
                contributions.append(hv_all - hv_without)
            worst_idx = int(np.argmin(contributions))
            pop.pop(worst_idx)
            F.pop(worst_idx)

        # optional logging
        if (gen + 1) % max(1, (n_gen // 10)) == 0:
            cur_best = np.min(F, axis=0)
            print(f"Gen {gen+1}/{n_gen}  pop_size={len(pop)}  best_f = {cur_best}  ref={ref_point}")

    return np.array(pop), np.array(F)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    resource_dir = "TTP_resoure"
    hpi_dir = "HPI"
    res_f_dir = "result_f"
    res_graph_dir = "result_graph"
    
    os.makedirs(res_f_dir, exist_ok=True)
    os.makedirs(res_graph_dir, exist_ok=True)

    # 1. Select Problem
    files = sorted(glob.glob(os.path.join(resource_dir, "*.txt")))
    # if not files: 
    #     print("No files found in TTP_resoure/ directory.")
    #     return

    print(f"\n{'='*40}\n Available TTP Problems \n{'='*40}")
    for i, f in enumerate(files): 
        print(f" [{i}] {os.path.basename(f)}")
        
    try:
        idx = int(input("Select problem index: "))
        selected_path = files[idx]
        file_name = os.path.basename(selected_path)
        base_name = file_name.replace(".txt", "")
    except Exception as e: 
        print(f"Invalid selection: {e}")
        # return

    # 2. Temp Load Data for ILS (Core Optimization)
    # We need a quick load just for coords and items to run the pre-optimization
    temp_problem_nodes = []
    temp_items = []
    
    with open(selected_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        start = False; item_sec = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"): start = True; continue
            if line.startswith("ITEMS SECTION"): start = False; item_sec = True; continue
            if start and line[0].isdigit():
                parts = line.split()
                temp_problem_nodes.append([float(parts[1]), float(parts[2])])
            if item_sec and line and line[0].isdigit():
                parts = line.split()
                temp_items.append([float(parts[1]), float(parts[2]), int(parts[3])-1])
                
    nodes_arr = np.array(temp_problem_nodes)
    items_arr = np.array(temp_items)
    
    # 3. Run Pre-Optimizers (ILS + Orientation)
    ils_time = 180 # Seconds allowed for ILS
    best_tour = run_parallel_ils(nodes_arr, time_limit=ils_time)
    
    # Critical Step: Orientation Check
    best_tour = optimize_tour_orientation(nodes_arr, best_tour, items_arr)
    problem = TTPProblem(selected_path, best_tour)
    adapter = TTPProblemAdapter(problem)   # if you already have the adapter; otherwise ensure problem.evaluate exists

    # DIAGNOSTICS (paste right after adapter = TTPProblemAdapter(problem))

    # 1) Eval consistency check on one random sampled individual
    x = np.random.rand(problem.n_cities + problem.n_items)
    out1 = {}
    problem.problem._evaluate(np.atleast_2d(x), out1) if hasattr(problem, "problem") else None  # guard if using adapter
    # Evaluate via original TTPProblem (use problem.problem if adapter else problem)
    try:
        if hasattr(problem, "problem"):
            vals_orig = problem.problem._evaluate(np.atleast_2d(x), {})  # we only want to ensure it runs
            # get F by calling underlying _evaluate
            out = {}
            problem.problem._evaluate(np.atleast_2d(x), out)
            f_original = out["F"][0]
        else:
            out = {}
            problem._evaluate(np.atleast_2d(x), out)
            f_original = out["F"][0]
    except Exception as e:
        print("Original _evaluate failed:", e)
        f_original = None

    f_adapter = problem.evaluate(x)
    print("Eval check — originalxw F (if available):", f_original, " adapter F:", f_adapter)

    # 2) Check whether CustomCrossover modifies tours
    pA = np.random.rand(problem.n_cities + problem.n_items)
    pB = np.random.rand(problem.n_cities + problem.n_items)
    # If using your provided CustomCrossover class
    try:
        cc = CustomCrossover(problem)  # create a fresh instance
        parents = np.vstack([pA, pB])  # shape (2, n_var)
        children = cc._do(problem, parents)
        print("Crossover produced children shape:", np.array(children).shape)
        # compare tours (first n_cities keys)
        print("parentA tour keys equal child1 tour keys?:", np.allclose(pA[:problem.n_cities], children[0][:problem.n_cities]))
        print("parentB tour keys equal child2 tour keys?:", np.allclose(pB[:problem.n_cities], children[1][:problem.n_cities]))
    except Exception as e:
        print("Crossover test failed:", e)


    pop, F = smsemoa_improved(
      adapter,   # or the original TTPProblem if it provides evaluate()
      n_var=problem.n_cities + problem.n_items,
      n_obj=2,
      pop_size=100,         # try 40 or 50 first
      n_gen=500,
      sampling=TunableSpectrumSampling(),  # optional
      crossover=CustomCrossover(problem),      # or pass your CustomCrossover(problem) if you want
      mutation=SmartMutation(problem),       # or SmartMutation(problem) if you prefer
      tour_mut_prob=0.3,
      pack_mut_prob=0.25,
      tournament_size=3,
      seed=42
    )

    # F = np.array(F)
    # sorted_idx = np.argsort(F[:,0])
    # F_sorted = F[sorted_idx]

    # 5. Save and Plot Results
    f_path = os.path.join(res_f_dir, f"RES_{base_name}.f")
    F = np.array(F)
    sorted_idx = np.argsort(F[:,0])
    F_sorted = F[sorted_idx]
    
    print(f"\n>>> Saving Results to {f_path}...")
    with open(f_path, 'w') as f_out:
        for row in F_sorted: 
            f_out.write(f"{row[0]:.2f} {-row[1]:.2f}\n")

    graph_path = os.path.join(res_graph_dir, f"PLOT_{base_name}.png")
    plt.figure(figsize=(12, 8))
    
    # Load HPI Reference if exists
    hpi_file = os.path.join(hpi_dir, f"HPI_{base_name}.f")
    if os.path.exists(hpi_file):
        try:
            hpi = np.loadtxt(hpi_file)
            if hpi.ndim > 1:
                # Ensure HPI is positive profit if formatted that way
                if hpi[0, 1] > 0: hpi[:, 1] = -hpi[:, 1]
                hpi = hpi[np.argsort(hpi[:, 0])]
                plt.plot(hpi[:, 0], hpi[:, 1], 'g--', linewidth=2.5, alpha=0.9, label='HPI Reference')
        except: pass

    # plt.scatter(F_init[:, 0], F_init[:, 1], c='gray', marker='x', s=40, alpha=0.4, label='Initial')
    plt.plot(F_sorted[:, 0], F_sorted[:, 1], 'r-o', linewidth=1.5, markersize=5, label='Optimized')
    plt.xlabel("Travel Time"); plt.ylabel("Negative Profit")
    plt.title(f"TTP Evolution: {base_name}")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(graph_path, dpi=150)
    print(f">>> Done. Graph saved to {graph_path}")
