import numpy as np
import time
import multiprocessing
import concurrent.futures
from scipy.spatial import cKDTree

# ==========================================
# 1. VECTORIZED LOCAL SEARCH (THE ENGINE)
# ==========================================
# These functions handle the geometry. We use NumPy vectorization to calculate thousands of distances at once.

def get_tour_length(tour, nodes):
    """ 
    Calculates the total Euclidean distance of the tour.
    It uses np.roll to pair node[i] with node[i+1] (and the last with the first)
    in a single matrix operation.
    """
    coords = nodes[tour]
    # Shift array by -1 to line up (x, y) with (next_x, next_y)
    diffs = coords - np.roll(coords, -1, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def full_deterministic_2opt(tour, nodes):
    """ 
    Deterministic 2-opt Scan.
    
    Logic:
    This looks at every single pair of edges in the tour and asks:
    "If I uncross these edges, does the tour get shorter?"
    """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 2):
            # Define the first edge (p1 -> p2)
            p1 = coords[i]
            p2 = coords[i+1]
            
            # Vectorized check against all other edges (p3 -> p4)
            j_indices = np.arange(i + 2, n)
            if len(j_indices) == 0: continue
            
            p3s = coords[j_indices]
            # p4 is the node after p3. Modulo n handles the wrap-around at the end.
            p4_indices = (j_indices + 1) % n
            p4s = coords[p4_indices]
            
            # Calculate current distance vs swapped distance for ALL j's at once
            d_curr = np.sqrt(np.sum((p1 - p2)**2)) + np.sqrt(np.sum((p3s - p4s)**2, axis=1))
            d_swap = np.sqrt(np.sum((p1 - p3s)**2, axis=1)) + np.sqrt(np.sum((p2 - p4s)**2, axis=1))
            
            diffs = d_swap - d_curr
            min_idx = np.argmin(diffs)
            
            # If the best swap is negative , apply it
            if diffs[min_idx] < -1e-6:
                j = j_indices[min_idx]
                # Reversing the segment performs the 'uncrossing'
                best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                coords = nodes[best_tour] # Update coordinate cache
                improved = True
                break # Restart scan after a modification
    return best_tour

def randomized_vectorized_2opt(tour, nodes, window=1000, max_scans=3000):
    """ 
    Stochastic Approximation Scan.
    
    Logic:
    For massive maps (e.g., 30,000 cities), checking every edge is impossible.
    This function picks random edges and only checks their NEIGHBORS (window)
    for swaps. 
    """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    
    # Don't scan more times than there are cities
    n_scans = min(max_scans, n)
    start_indices = np.random.randint(0, n - 3, size=n_scans)
    
    for i in start_indices:
        p1 = coords[i]; p2 = coords[i+1]
        
        # Only look 'window' steps ahead, not the whole array
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
    return best_tour

def double_bridge_kick(tour):
    """
    The Perturbation Mechanism.
    
    Logic:
    2-opt gets stuck in "Local Minima". To get out, we need to 
    break the tour significantly. This function slices the tour into 4 pieces
    and reconnects them in a specific non-sequential order. 
    
    It's like shuffling keeps most structure 
    but changes the connections enough to allow 2-opt to find new improvements.
    """
    n = len(tour)
    if n < 8: return tour
    # Pick 4 random cut points
    pos = np.sort(np.random.choice(range(1, n-1), 4, replace=False))
    # Reassemble: A-D-C-B-E
    return np.concatenate((tour[:pos[0]], tour[pos[2]:pos[3]], tour[pos[1]:pos[2]], tour[pos[0]:pos[1]], tour[pos[3]:]))

# ==========================================
# 2. ADAPTIVE ILS WORKER 
# ==========================================
def ils_worker(args):
    """
    Runs a complete Iterated Local Search (ILS) process on a single CPU core.
    """
    nodes, seed, time_limit = args

    # Ensure this process has a unique random seed
    np.random.seed(seed)
    n = len(nodes)
    start_time = time.time()
    
    # --- PHASE 1: INITIALIZATION (Nearest Neighbor) ---
    # We don't start random. We start with a path using a KD-Tree.
    # This gives us a decent starting score immediately.
    start_node = np.random.randint(0, n)
    tree = cKDTree(nodes) # Spatial index for fast lookup
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)
    
    curr = start_node; tour[0] = curr; visited[curr] = True
    
    # Optimization: Only look at the closest 100 neighbors, fallback to global if trapped
    k_search = 100 if n > 10000 else n
    
    for i in range(1, n):
        dists, indices = tree.query(nodes[curr], k=k_search)
        if np.isscalar(indices): indices = [indices]
        found = False
        for idx in indices:
            if not visited[idx]:
                tour[i] = idx; visited[idx] = True; curr = idx; found = True; break
        # Fallback: if all near neighbors are visited, pick the first available node
        if not found:
            rem = np.where(~visited)[0]
            if len(rem) > 0:
                idx = rem[0]; tour[i] = idx; visited[idx] = True; curr = idx

    best_tour = tour.copy()
    best_len = get_tour_length(best_tour, nodes)
    
    # Adaptive Logic: If map is small, use precise search. If huge, use fast search.
    use_full_scan = (n < 2000)
    
    # --- PHASE 2: ITERATED LOCAL SEARCH LOOP ---
    while (time.time() - start_time) < time_limit:
        
        # A. Local Search (Hill Climbing)
        if use_full_scan:
            best_tour = full_deterministic_2opt(best_tour, nodes)
        else:
            best_tour = randomized_vectorized_2opt(best_tour, nodes, window=2000)
            
        curr_len = get_tour_length(best_tour, nodes)
        if curr_len < best_len: best_len = curr_len
        
        # B. Perturbation (The Kick)
        # We deliberately break the tour to jump out of the local minimum
        kicked = double_bridge_kick(best_tour)
        
        # C. Repair
        # We optimize the broken tour to see if it settles into a BETTER minimum
        if use_full_scan:
            repaired = full_deterministic_2opt(kicked, nodes)
        else:
            repaired = randomized_vectorized_2opt(kicked, nodes, window=1500)
            
        d = get_tour_length(repaired, nodes)
        
        # D. Acceptance Criterion
        # If it shorter distance, we keep it.
        # Otherwise, we discard 'repaired' and go back to 'best_tour' for a different kick.
        if d < best_len:
            best_len = d; best_tour = repaired
            
    return best_tour, best_len

def run_parallel_ils(nodes, time_limit=30, seed=None):
    """
    Orchestrator: Runs multiple ILS instances on all available CPU cores.
    Since ILS is stochastic (randomized), running 8 times increases the odds
    of finding the 'Global Optimum'.
    """
    print(f"   > Starting Parallel ILS ({len(nodes)} cities) | Limit: {time_limit}s...")

    if seed is not None:
        np.random.seed(seed)
    
    # Leave one core free so the OS doesn't freeze
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create unique seeds so workers don't do the exact same thing
    tasks = [(nodes, np.random.randint(0, 1000000) + i, time_limit) for i in range(n_workers)]
    
    best_global_tour = None; best_global_len = float('inf')
    
    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(ils_worker, tasks)
        for tour, score in results:
            print(f"     Worker Finished | Score: {score:.1f}")
            if score < best_global_len: best_global_len = score; best_global_tour = tour
            
    print(f"   > Best Tour Found: {best_global_len:.1f}")
    return best_global_tour

# ==========================================
# 3. TOUR ORIENTATION (THE THIEF STRATEGY)
# ==========================================
def optimize_tour_orientation(nodes, tour, items):
    """
    Determines if we should drive the route Clockwise or Counter-Clockwise.
    
    The Logic (Physics of TTP):
    - In the Traveling Thief Problem, the knapsack gets heavier as you pick items.
    - Heavier knapsack = Slower travel speed.
    - Speed formula is usually: v = v_max - weight * drop_rate
    
    Strategy:
    You want to travel the LONGEST distances while your bag is EMPTY.
    You want to pick up the HEAVIEST items right before you finish.
    
    This function calculates a score based on (Weight * Distance_From_Start)
    and flips the tour if the reverse direction is more efficient.
    """
    # Normalize tour to start at node 0 (the depot)
    start_node_idx = np.where(tour == 0)[0][0]
    aligned_tour = np.roll(tour, -start_node_idx)
    
    # Create the reverse version (0 -> N -> N-1 ... -> 1)
    reverse_tour = np.concatenate(([aligned_tour[0]], aligned_tour[1:][::-1]))
    
    def get_score(t):
        # Calculate cumulative distance from start for every node in the sequence
        coords = nodes[t]
        dists = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        cum_dist = np.cumsum(dists)[:-1] 
        cum_dist = np.insert(cum_dist, 0, 0)
        
        # Map node indices to their distance from start
        dist_map = np.zeros(len(nodes))
        dist_map[t] = cum_dist
        
        item_locs = items[:, 2].astype(int)
        item_ws = items[:, 1]
        
        # Score = Sum(ItemWeight * DistanceFromStart)
        # High Score means heavy items are far from start (picked up late).
        # We WANT a high score here (maximize distance traveled light).
        score = np.sum(item_ws * dist_map[item_locs])
        return score

    print("   > Optimizing Tour Direction...")
    s_fwd = get_score(aligned_tour)
    s_rev = get_score(reverse_tour)
    
    if s_rev > s_fwd:
        print("     -> REVERSING tour. Logic: Heavy items found later in this direction.")
        return reverse_tour
    else:
        print("     -> Forward tour is optimal.")
        return aligned_tour