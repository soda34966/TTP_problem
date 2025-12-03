import numpy as np

class TTPProblemStandalone:
    def __init__(self, filename, base_tour):
        self.filename = filename
        self.base_tour = base_tour
        self._load_data(filename)
        self._precalc_physics()
    
    def _load_data(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        self.nodes, self.items = [], []
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
        self.nodes = np.array(self.nodes)
        self.items = np.array(self.items)
        self.n_cities = len(self.nodes)
        self.n_items = len(self.items)
        self.item_locs = self.items[:, 2].astype(int)
        self.item_weights = self.items[:, 1]
        self.item_profits = self.items[:, 0]
    
    def _precalc_physics(self):
        tour = self.base_tour
        coords = self.nodes[tour]
        edges = np.sqrt(np.sum((coords - np.roll(coords, -1, axis=0))**2, axis=1))
        dist_map = np.zeros(self.n_cities)
        cum_dist = 0
        for i in range(self.n_cities-1, -1, -1):
            dist_map[tour[i]] = cum_dist
            cum_dist += edges[i]
        dists = dist_map[self.item_locs]
        cost = self.item_weights * dists
        if cost.max() > cost.min():
            cost = (cost - cost.min()) / (cost.max() - cost.min())
        self.physics_scores = self.item_profits / (cost + 1e-6)
        self.ratio_scores = self.item_profits / (self.item_weights + 1e-9)

    def evaluate(self, x):
        """
        x: array of length n_cities + n_items
        returns: 2-objective vector [total_time, -total_profit]
        """
        tour = np.argsort(x[:self.n_cities])
        pack_genes = x[self.n_cities:]
        picked_mask = pack_genes > 0.5
        
        # Greedy repair for capacity
        current_w = np.sum(self.item_weights[picked_mask])
        if current_w > self.capacity:
            idxs = np.where(picked_mask)[0]
            sorted_idxs = idxs[np.argsort(self.ratio_scores[idxs])]
            ws = self.item_weights[sorted_idxs]
            cum_drop = np.cumsum(ws)
            excess = current_w - self.capacity
            drop_count = np.searchsorted(cum_drop, excess) + 1
            picked_mask[sorted_idxs[:drop_count]] = False
        
        current_p = np.sum(self.item_profits[picked_mask])
        city_weights = np.zeros(self.n_cities)
        if np.any(picked_mask):
            np.add.at(city_weights, self.item_locs[picked_mask], self.item_weights[picked_mask])
        
        coords_ordered = self.nodes[tour]
        weights_ordered = city_weights[tour]
        dists = np.sqrt(np.sum((coords_ordered - np.roll(coords_ordered, -1, axis=0))**2, axis=1))
        cur_loads = np.cumsum(weights_ordered)
        velocities = np.clip(self.v_max - (cur_loads / self.capacity)*(self.v_max - self.v_min), self.v_min, self.v_max)
        total_time = np.sum(dists / velocities)
        return np.array([total_time, -current_p])

class TTPProblemAdapter:
    """
    Wrap TTPProblem for the pure Python SMS-EMOA
    """
    def __init__(self, problem):
        self.problem = problem
        self.n_var = problem.n_cities + problem.n_items
        # Expose attributes needed by sampling, mutation, crossover
        self.n_cities = problem.n_cities
        self.nodes = problem.nodes
        self.base_tour = problem.base_tour
        self.item_locs = problem.item_locs
        self.item_weights = problem.item_weights
        self.item_profits = problem.item_profits
        self.capacity = problem.capacity
        self.physics_scores = problem.physics_scores

    def evaluate(self, x):
        """
        x: 1D array of length n_cities + n_items
        Returns: 2-objective np.array [travel_time, -profit]
        """
        X = np.atleast_2d(x)   # _evaluate expects batch input
        out = {}
        self.problem._evaluate(X, out)
        return out["F"][0]      # return 1D array for this candidate
