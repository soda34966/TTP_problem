import numpy as np
import time
import multiprocessing
import concurrent.futures
from scipy.spatial import cKDTree

# ==========================================
# 1.VECTORIZED 
# ==========================================
def get_tour_length(tour, nodes):
    coords = nodes[tour]
    diffs = coords - np.roll(coords, -1, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def full_deterministic_2opt(tour, nodes):
    """ Deterministic Scan. """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 2):
            p1 = coords[i]
            p2 = coords[i+1]
            j_indices = np.arange(i + 2, n)
            if len(j_indices) == 0: continue
            
            p3s = coords[j_indices]
            p4_indices = (j_indices + 1) % n
            p4s = coords[p4_indices]
            
            d_curr = np.sqrt(np.sum((p1 - p2)**2)) + np.sqrt(np.sum((p3s - p4s)**2, axis=1))
            d_swap = np.sqrt(np.sum((p1 - p3s)**2, axis=1)) + np.sqrt(np.sum((p2 - p4s)**2, axis=1))
            
            diffs = d_swap - d_curr
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < -1e-6:
                j = j_indices[min_idx]
                best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                coords = nodes[best_tour]
                improved = True
                break 
    return best_tour

def randomized_vectorized_2opt(tour, nodes, window=1000, max_scans=3000):
    """ Fast Approximation Scan for N > 2000. """
    n = len(tour)
    best_tour = tour.copy()
    coords = nodes[best_tour]
    n_scans = min(max_scans, n)
    start_indices = np.random.randint(0, n - 3, size=n_scans)
    
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
    
    use_full_scan = (n < 2000)
    
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
# 3. TOUR ORIENTATION OPTIMIZER
# ==========================================
def optimize_tour_orientation(nodes, tour, items):
    """
    Determines if the tour should be traversed Forward or Backward.
    Heuristic: Heavy items should be visited LATE (high distance from start).
    """
    start_node_idx = np.where(tour == 0)[0][0]
    aligned_tour = np.roll(tour, -start_node_idx)
    
    reverse_tour = np.concatenate(([aligned_tour[0]], aligned_tour[1:][::-1]))
    
    def get_score(t):
        coords = nodes[t]
        dists = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        cum_dist = np.cumsum(dists)[:-1] 
        cum_dist = np.insert(cum_dist, 0, 0)
        
        dist_map = np.zeros(len(nodes))
        dist_map[t] = cum_dist
        
        item_locs = items[:, 2].astype(int)
        item_ws = items[:, 1]
        
        # heavy items with HIGH cumulative distance (picked up last)
        score = np.sum(item_ws * dist_map[item_locs])
        return score

    print("   > Optimizing Tour Direction...")
    s_fwd = get_score(aligned_tour)
    s_rev = get_score(reverse_tour)
    
    if s_rev > s_fwd:
        print("     -> REVERSING tour for better weight distribution.")
        return reverse_tour
    else:
        print("     -> Forward tour is optimal.")
        return aligned_tour