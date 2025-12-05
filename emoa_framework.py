import numpy as np
import copy

# ==========================================
# EMOA CLASSES
# ==========================================

class Problem:
    def __init__(self, n_var, n_obj, xl, xu):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu

    def evaluate(self, X):
        raise NotImplementedError

class Individual:
    def __init__(self, X=None):
        self.X = X
        self.F = None
        self.rank = None
        self.crowding_dist = None
        self.hv_contrib = 0.0

class Population:
    def __init__(self, individuals=None):
        self.individuals = individuals if individuals is not None else []

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def append(self, ind):
        self.individuals.append(ind)

    def get_X(self):
        return np.array([ind.X for ind in self.individuals])

    def get_F(self):
        return np.array([ind.F for ind in self.individuals])

    def set_F(self, F):
        for i, ind in enumerate(self.individuals):
            ind.F = F[i]

# ==========================================

def fast_non_dominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n)
    rank = np.zeros(n)
    fronts = [[]]

    for p in range(n):
        S[p] = []
        n_dom[p] = 0
        for q in range(n):
            if p == q: continue
            if all(F[p] <= F[q]) and any(F[p] < F[q]):
                S[p].append(q)
            elif all(F[q] <= F[p]) and any(F[q] < F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break 
    return fronts

def calc_2d_hypervolume_contribution(F_front, ref_point):
    # Sort by first objective
    sorted_indices = np.argsort(F_front[:, 0])
    F_sorted = F_front[sorted_indices]
    n = len(F_sorted)
    contributions = np.zeros(n)

    for i in range(n):
        # Calculate area of exclusive rectangle
        next_f1 = F_sorted[i+1, 0] if i < n - 1 else ref_point[0]
        prev_f2 = F_sorted[i-1, 1] if i > 0 else ref_point[1]
        
        width = next_f1 - F_sorted[i, 0]
        height = prev_f2 - F_sorted[i, 1]
        contributions[i] = width * height

    original_contributions = np.zeros(n)
    original_contributions[sorted_indices] = contributions
    return original_contributions

class CustomSMSEMOA:
    def __init__(self, problem, pop_size, sampling, crossover, mutation):
        self.problem = problem
        self.pop_size = pop_size
        self.sampling = sampling
        self.crossover = crossover
        self.mutation = mutation
        self.pop = None
        self.initial_F = None 

    def initialize(self):
        X_init = self.sampling.do(self.problem, self.pop_size)
        self.pop = Population([Individual(x) for x in X_init])
        
        F_init = self.problem.evaluate(self.pop.get_X())
        self.pop.set_F(F_init)
        self.initial_F = F_init.copy()

    def run(self, n_gen, verbose=True):
        self.initialize()

        for gen in range(n_gen):
            X_parents = self.pop.get_X()
            indices = np.arange(self.pop_size)
            np.random.shuffle(indices)
            
            offspring_list = []
            for i in range(0, self.pop_size, 2):
                p1_idx, p2_idx = indices[i], indices[(i+1)%self.pop_size]
                p1, p2 = X_parents[p1_idx], X_parents[p2_idx]
                
                kids_X = self.crossover.do(p1, p2)
                for k in range(len(kids_X)):
                    kids_X[k] = self.mutation.do(kids_X[k])
                    offspring_list.append(Individual(kids_X[k]))
            
            off_pop = Population(offspring_list)
            F_off = self.problem.evaluate(off_pop.get_X())
            off_pop.set_F(F_off)
            
            merged_inds = self.pop.individuals + off_pop.individuals
            all_F = np.array([ind.F for ind in merged_inds])
            
            # --- Normalization ---
            ideal_point = np.min(all_F, axis=0)
            nadir_point = np.max(all_F, axis=0)
            denom = nadir_point - ideal_point
            denom[denom < 1e-6] = 1.0
            norm_F = (all_F - ideal_point) / denom

            fronts = fast_non_dominated_sort(all_F)
            new_inds = []
            remaining_slots = self.pop_size
            
            for front_indices in fronts:
                if len(front_indices) <= remaining_slots:
                    for idx in front_indices:
                        new_inds.append(merged_inds[idx])
                    remaining_slots -= len(front_indices)
                else:
                    current_front_indices = list(front_indices)
                    
                    ref_point = np.ones(2) + 1.0 
                    
                    while len(current_front_indices) > remaining_slots:
                        sub_F_norm = norm_F[current_front_indices]
                        hv_contribs = calc_2d_hypervolume_contribution(sub_F_norm, ref_point)
                        worst_idx_local = np.argmin(hv_contribs)
                        del current_front_indices[worst_idx_local]

                    for idx in current_front_indices:
                        new_inds.append(merged_inds[idx])
                    remaining_slots = 0
                
                if remaining_slots == 0:
                    break
            
            self.pop = Population(new_inds)
            
            if verbose and (gen % 10 == 0 or gen == n_gen - 1):
                best_f1 = np.min([ind.F[0] for ind in self.pop])
                best_f2 = np.min([ind.F[1] for ind in self.pop])
                print(f"Gen {gen+1}/{n_gen} | Front: {len(fronts[0])} | Best T: {best_f1:.0f} | Best P: {-best_f2:.0f}")

        return self.pop