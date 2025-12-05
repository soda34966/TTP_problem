import numpy as np

# ==========================================
# 1. SAMPLING
# ==========================================
class TunableSpectrumSampling:
    def __init__(self, sigma_min=1.0, sigma_max=10.0, seed=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rng = np.random.default_rng(seed)

    def do(self, problem, n_samples):
        n_var = problem.n_var
        n_c = problem.n_cities
        
        X = np.zeros((n_samples, n_var))
        
        base_tour = problem.base_tour
        perfect_keys = np.linspace(0.00001, 0.99999, n_c)
        base_k = np.zeros(n_c)
        for rank, city_idx in enumerate(base_tour):
            base_k[city_idx] = perfect_keys[rank]

        coords = problem.nodes[base_tour]
        edges = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        dist_map = np.zeros(n_c); cum_dist = 0
        for i in range(n_c - 1, -1, -1):
            dist_map[base_tour[i]] = cum_dist; cum_dist += edges[i]
        item_dists = dist_map[problem.item_locs]

        for i in range(n_samples):
            # --- A. Tour Component ---
            if i < 2: 
                X[i, :n_c] = base_k
            else:
                step_size = 1.0 / n_c
                # Use Instance RNG
                sigma = step_size * self.rng.uniform(self.sigma_min, self.sigma_max) 
                noise = self.rng.normal(0, sigma, n_c)
                X[i, :n_c] = np.clip(base_k + noise, 0, 1)

            # --- B. PACKING COMPONENT ---
            if i == 0:
                X[i, n_c:] = 0.0
            elif i == 1:
                ratios = problem.item_profits / (problem.item_weights + 1e-9)
                sorted_items = np.argsort(-ratios)
                curr_w = 0
                for idx in sorted_items:
                    if curr_w + problem.item_weights[idx] <= problem.capacity:
                        X[i, n_c + idx] = 1.0
                        curr_w += problem.item_weights[idx]
            else:
                if self.rng.random() < 0.3:
                    X[i, n_c:] = (self.rng.random(problem.n_items) < 0.1).astype(float)
                else:
                    target_pct = (i - 1) / (n_samples - 2) 
                    target_weight = problem.capacity * target_pct
                    alpha = self.rng.uniform(0.5, 1.5)
                    tuned_cost = problem.item_weights * (item_dists ** alpha + 1e-9)
                    tuned_scores = problem.item_profits / (tuned_cost + 1e-9)
                    
                    ranked_items = np.argsort(-tuned_scores)
                    current_w = 0
                    for idx in ranked_items:
                        w = problem.item_weights[idx]
                        if current_w + w <= target_weight:
                            X[i, n_c + idx] = 1.0
                            current_w += w
        return X

# ==========================================
# 2. MUTATION And CROSSOVER
# ==========================================
class Mutation:
    def __init__(self, problem, config=None, seed=None):
        self.n_cities = problem.n_cities
        self.n_items = problem.n_items
        self.nodes = problem.nodes
        self.scores = problem.physics_scores
        self.tolerance = 500.0 
        
        # Load Defaults or Config
        if config is None:
            config = {}
        
        self.pack_prob = config.get('mut_pack_prob', 0.5)
        self.tour_prob = config.get('mut_tour_prob', 0.3)
        self.flip_prob = config.get('mut_pack_flip_prob', 0.1)
        self.purge_prob = config.get('mut_pack_purge_prob', 0.15)
        self.bridge_prob = config.get('mut_tour_bridge_prob', 0.15)
        

        _seed = seed if seed is not None else config.get('seed', None)
        self.rng = np.random.default_rng(_seed)

    def do(self, x_ind):
        x_mut = x_ind.copy()
        
        # 1. KNAPSACK MUTATION
        if self.rng.random() < self.pack_prob:
            pack_genes = x_mut[self.n_cities:]
            rand_val = self.rng.random()
            
            # Mode A: Random Flip (Exploration)
            if rand_val < self.flip_prob:
                n_flips = max(1, int(self.n_items * 0.05))
                idxs = self.rng.choice(self.n_items, n_flips, replace=False)
                pack_genes[idxs] = 1.0 - pack_genes[idxs]
            
            # Mode B: The "Purge" (Reset to empty)
            elif rand_val < (self.flip_prob + self.purge_prob):
                pack_genes[:] = 0.0
                
            # Mode C: Standard Bit Flip (Refinement)
            else:
                n_flips = self.rng.integers(1, 5)
                idxs = self.rng.choice(self.n_items, n_flips, replace=False)
                pack_genes[idxs] = 1.0 - pack_genes[idxs]

            x_mut[self.n_cities:] = pack_genes

        # 2. TOUR MUTATION
        if self.rng.random() < self.tour_prob:
            # Mode A: Double Bridge (Large Jump)
            if self.rng.random() < self.bridge_prob:
                keys = x_mut[:self.n_cities]
                n = len(keys)
                if n >= 8:
                    idxs = np.sort(self.rng.choice(n, 4, replace=False))
                    keys[idxs[0]:idxs[1]] += 0.8 
                    keys[idxs[2]:idxs[3]] -= 0.8
                    x_mut[:self.n_cities] = np.clip(keys, 0, 1)
            
            # Mode B: 2-Opt (Local Search)
            else:
                tour_keys = x_mut[:self.n_cities]
                current_order = np.argsort(tour_keys)
                
                a = self.rng.integers(1, self.n_cities - 2)
                window = self.rng.integers(2, 50)
                b = min(a + window, self.n_cities - 1)
                
                idx_prev, idx_a = current_order[a-1], current_order[a]
                idx_b, idx_next = current_order[b], current_order[(b+1) % self.n_cities]
                
                p_prev, p_a = self.nodes[idx_prev], self.nodes[idx_a]
                p_b, p_next = self.nodes[idx_b], self.nodes[idx_next]
                
                d_curr = np.linalg.norm(p_prev - p_a) + np.linalg.norm(p_b - p_next)
                d_new  = np.linalg.norm(p_prev - p_b) + np.linalg.norm(p_a - p_next)
                
                if (d_new - d_curr) < self.tolerance:
                    key_indices = current_order[a:b+1]
                    keys_values = x_mut[key_indices]
                    x_mut[key_indices] = keys_values[::-1] 
                
        return x_mut

class Crossover:
    def __init__(self, problem, seed=None):
        self.n_cities = problem.n_cities
        self.rng = np.random.default_rng(seed)

    def do(self, p1, p2):
        off1, off2 = p1.copy(), p2.copy()
        
        # Packing: Uniform Crossover
        pack_start = self.n_cities
        pack1, pack2 = p1[pack_start:], p2[pack_start:]
        mask = self.rng.random(len(pack1)) < 0.5
        off1[pack_start:] = np.where(mask, pack1, pack2)
        off2[pack_start:] = np.where(mask, pack2, pack1)
        
        return [off1, off2]

class TTPProblem:
    def __init__(self, filename, base_tour):
        self.filename = filename
        self.base_tour = base_tour 
        self._load_data(filename)
        self._precalc_physics() 
        self.n_var = self.n_cities + self.n_items
        self.n_obj = 2
        self.xl = 0
        self.xu = 1

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

    def evaluate(self, X):
        n_pop = len(X); F = np.zeros((n_pop, 2)); cap = self.capacity; v_diff = self.v_max - self.v_min
        item_w = self.item_weights; item_p = self.item_profits
        item_l = self.item_locs; r_scores = self.ratio_scores; nodes = self.nodes
        
        for i in range(n_pop):
            tour = np.argsort(X[i, :self.n_cities])
            pack_genes = X[i, self.n_cities:]
            picked_mask = pack_genes > 0.5
            
            current_w = np.sum(item_w[picked_mask])
            if current_w > cap:
                idxs = np.where(picked_mask)[0]
                sorted_idxs = idxs[np.argsort(r_scores[idxs])]
                ws = item_w[sorted_idxs]; cum_drop = np.cumsum(ws)
                excess = current_w - cap
                drop_count = np.searchsorted(cum_drop, excess) + 1
                picked_mask[sorted_idxs[:drop_count]] = False
            
            current_p = np.sum(item_p[picked_mask])
            
            city_weights = np.zeros(self.n_cities)
            if np.any(picked_mask): 
                np.add.at(city_weights, item_l[picked_mask], item_w[picked_mask])
            
            coords_ordered = nodes[tour]
            weights_ordered = city_weights[tour]
            
            dists = np.sqrt(np.sum((coords_ordered - np.roll(coords_ordered, -1, axis=0))**2, axis=1))
            cur_loads = np.cumsum(weights_ordered)
            velocities = np.clip(self.v_max - (cur_loads / cap) * v_diff, self.v_min, self.v_max)
            
            F[i, 0] = np.sum(dists / velocities)
            F[i, 1] = -current_p 
            
        return F