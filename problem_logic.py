import numpy as np

# ==========================================
# 1. TUNABLE SPECTRUM SAMPLING
# ==========================================
class TunableSpectrumSampling:
    def do(self, problem, n_samples):
        # Access problem attributes directly
        n_var = problem.n_var
        n_c = problem.n_cities
        
        X = np.zeros((n_samples, n_var))
        
        # 1. Setup Base Keys based on ILS Tour
        base_tour = problem.base_tour
        # Create keys such that argsort(keys) == base_tour
        perfect_keys = np.linspace(0.00001, 0.99999, n_c)
        
        # keys[base_tour[i]] < keys[base_tour[i+1]]
        base_k = np.zeros(n_c)
        for rank, city_idx in enumerate(base_tour):
            base_k[city_idx] = perfect_keys[rank]

        # 2. Pre-calc distances
        coords = problem.nodes[base_tour]
        edges = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        dist_map = np.zeros(n_c); cum_dist = 0
        for i in range(n_c - 1, -1, -1):
            dist_map[base_tour[i]] = cum_dist; cum_dist += edges[i]
        item_dists = dist_map[problem.item_locs]

        # 3. Generate Population
        for i in range(n_samples):
            # --- A. The Gaussian Cloud ---
            if i == 0:
                X[i, :n_c] = base_k
            else:
                step_size = 1.0 / n_c
                # Wider sigma for exploration
                sigma = step_size * np.random.uniform(1.0, 5.0) 
                noise = np.random.normal(0, sigma, n_c)
                X[i, :n_c] = np.clip(base_k + noise, 0, 1)

            # --- B. PACKING COMPONENT ---
            target_pct = i / (n_samples - 1)
            target_weight = problem.capacity * target_pct
            alpha = 1.0 - (target_pct * 0.8)
            
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
# 2. CUSTOM OPERATORS
# ==========================================
class Mutation:
    def __init__(self, problem):
        self.n_cities = problem.n_cities
        self.nodes = problem.nodes
        self.pack_prob = 0.3
        self.tour_prob = 0.2
        self.scores = problem.physics_scores
        self.tolerance = 50.0 

    def do(self, x_ind):
        # x_ind is a 1D numpy array 
        x_mut = x_ind.copy()
        
        # 1. KNAPSACK MUTATION
        if np.random.random() < self.pack_prob:
            pack_genes = x_mut[self.n_cities:]
            picked = pack_genes > 0.5
            if np.any(picked):
                indices = np.where(picked)[0]
                # Add noise to scores to vary selection
                noisy_scores = self.scores[indices] * np.random.uniform(0.9, 1.1, len(indices))
                worst = indices[np.argmin(noisy_scores)]
                x_mut[self.n_cities + worst] = 0.0
            
            not_picked = ~picked
            if np.any(not_picked):
                indices = np.where(not_picked)[0]
                # Pick from top candidates
                candidates = indices[np.argsort(self.scores[indices])[-20:]]
                if len(candidates) > 0:
                    best = np.random.choice(candidates)
                    x_mut[self.n_cities + best] = 1.0

        # 2. GUARDED TOUR MUTATION
        if np.random.random() < self.tour_prob:
            tour_keys = x_mut[:self.n_cities]
            current_order = np.argsort(tour_keys)
            
            a = np.random.randint(1, self.n_cities - 2)
            window = np.random.randint(2, 100)
            b = min(a + window, self.n_cities - 1)
            
            # Indices in the tour array
            idx_prev = current_order[a-1]
            idx_a    = current_order[a]
            idx_b    = current_order[b]
            idx_next = current_order[(b+1) % self.n_cities] 
            
            p_prev = self.nodes[idx_prev]
            p_a    = self.nodes[idx_a]
            p_b    = self.nodes[idx_b]
            p_next = self.nodes[idx_next]
            
            d_current = np.linalg.norm(p_prev - p_a) + np.linalg.norm(p_b - p_next)
            d_new     = np.linalg.norm(p_prev - p_b) + np.linalg.norm(p_a - p_next)
            
            delta = d_new - d_current
            # Only apply if penalty is low
            if delta < self.tolerance:
                key_indices = current_order[a:b+1]
                keys_values = x_mut[key_indices]
                x_mut[key_indices] = keys_values[::-1] # Reverse values -> Reverse sort order
                
        return x_mut

class Crossover:
    def __init__(self, problem):
        self.n_cities = problem.n_cities

    def do(self, p1, p2):
        # p1, p2 are 1D arrays
        # Returns 2 offspring
        off1 = p1.copy()
        off2 = p2.copy()
        
        # Uniform crossover for bitmask
        mask = np.random.random(self.n_cities + len(p1[self.n_cities:])) < 0.5
        
        # Simplified: Uniform Crossover on Packing only
        pack_start = self.n_cities
        
        pack1 = p1[pack_start:]
        pack2 = p2[pack_start:]
        
        mask = np.random.random(len(pack1)) < 0.5
        
        new_pack1 = np.where(mask, pack1, pack2)
        new_pack2 = np.where(mask, pack2, pack1)
        
        off1[pack_start:] = new_pack1
        off2[pack_start:] = new_pack2
        
        # Keep tours from parents directly (Locked Tour) or could interpolate keys
        
        return [off1, off2]

# ==========================================
# 3. PROBLEM CLASS
# ==========================================
class TTPProblem:
    def __init__(self, filename, base_tour):
        self.filename = filename
        self.base_tour = base_tour 
        self._load_data(filename)
        self._precalc_physics() 
        
        # Attributes required by framework
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
        # X is (N_pop, n_var)
        n_pop = len(X); F = np.zeros((n_pop, 2)); cap = self.capacity; v_diff = self.v_max - self.v_min
        item_w = self.item_weights; item_p = self.item_profits
        item_l = self.item_locs; r_scores = self.ratio_scores; nodes = self.nodes
        
        for i in range(n_pop):
            # Decode Tour
            tour = np.argsort(X[i, :self.n_cities])
            
            # Decode Packing
            pack_genes = X[i, self.n_cities:]
            picked_mask = pack_genes > 0.5
            
            # Greedy Repair()
            current_w = np.sum(item_w[picked_mask])
            if current_w > cap:
                idxs = np.where(picked_mask)[0]
                # Drop items with worst Profit/Weight ratio 
                sorted_idxs = idxs[np.argsort(r_scores[idxs])]
                ws = item_w[sorted_idxs]; cum_drop = np.cumsum(ws)
                # Find how many to drop
                excess = current_w - cap
                drop_count = np.searchsorted(cum_drop, excess) + 1
                picked_mask[sorted_idxs[:drop_count]] = False
            
            current_p = np.sum(item_p[picked_mask])
            
            # TTP Travel Time Calculation
            city_weights = np.zeros(self.n_cities)
            if np.any(picked_mask): 
                np.add.at(city_weights, item_l[picked_mask], item_w[picked_mask])
            
            coords_ordered = nodes[tour]
            weights_ordered = city_weights[tour]
            
            dists = np.sqrt(np.sum((coords_ordered - np.roll(coords_ordered, -1, axis=0))**2, axis=1))
            # Load accumulation along the path
            cur_loads = np.cumsum(weights_ordered)
            # Velocity at each step. Note: TTP usually defines velocity based on weight *after* leaving city i.
            # Formula: v = v_max - (current_load / capacity) * (v_max - v_min)
            velocities = np.clip(self.v_max - (cur_loads / cap) * v_diff, self.v_min, self.v_max)
            
            # Distance is typically edge i -> i+1.
            # Standard TTP: Time = sum( dist[i] / v[i] )
            # The load on the return leg is the total load.
            
            F[i, 0] = np.sum(dists / velocities)
            F[i, 1] = -current_p # Minimize negative profit
            
        return F