from Survival import LeastHypervolumeContributionSurvival
import numpy as np
from copy import deepcopy


# -------------------------------
# Dominance check
# -------------------------------
def dominates(a, b):
    """Returns True if a dominates b"""
    return np.all(a <= b) and np.any(a < b)

# -------------------------------
# Hypervolume (2D)
# -------------------------------
def hv_2d(points, ref_point):
    """Compute hypervolume for 2 objectives"""
    # Sort by f1 ascending
    sorted_pts = sorted(points, key=lambda x: x[0])
    hv = 0.0
    prev_f1 = ref_point[0]
    for f1, f2 in sorted_pts:
        width = prev_f1 - f1
        height = ref_point[1] - f2
        hv += width * height
        prev_f1 = f1
    return hv

def tournament_advanced(pop, F, tour_size=3, ref_point=None):
    idxs = np.random.choice(len(pop), tour_size, replace=False)
    f_cand = [F[i] for i in idxs]
    # identify non-dominated among candidates
    def dominates(a,b): return np.all(a<=b) and np.any(a<b)
    nd = []
    for j, fj in enumerate(f_cand):
        if not any(dominates(fk, fj) for k,fk in enumerate(f_cand) if k!=j):
            nd.append((idxs[j], fj))
    if len(nd)==1: return pop[nd[0][0]]
    # tie-breaker: HV contribution
    if ref_point is None: ref_point = np.max(F,axis=0)*1.1
    f_nd = [x[1] for x in nd]
    # compute hv contributions among nd set
    contribs = []
    for i in range(len(f_nd)):
        others = [f for j,f in enumerate(f_nd) if j!=i]
        full = hv_2d(f_nd, ref_point)
        contribs.append(full - hv_2d(others, ref_point))
    best = nd[np.argmax(contribs)][0]
    return pop[best]



# -------------------------------
# SMS-EMOA main loop
# -------------------------------
def smsemoa(problem, n_var, n_obj, pop_size=100, n_gen=100,
            lower=0.0, upper=1.0,
            sampler=None, crossover=None, mutation=None):
    """
    Pure Python SMS-EMOA with optional sampling, crossover, and mutation.
    """

    # --- Initialize population ---
    if sampler is not None:
        X_init = sampler._do(problem, pop_size)
        pop = [X_init[i].copy() for i in range(pop_size)]
    else:
        pop = [np.random.uniform(lower, upper, n_var) for _ in range(pop_size)]

    # Evaluate initial population
    F = [problem.evaluate(ind) for ind in pop]

    # Reference point for hypervolume
    ref_point = np.max(F, axis=0) + 1.0

    for gen in range(n_gen):
        # # --- Selection (binary tournament) ---
        def tournament(pop, F):
            idx1, idx2 = np.random.choice(len(pop), 2, replace=False)
            if np.all(F[idx1] <= F[idx2]) and np.any(F[idx1] < F[idx2]):
                return pop[idx1]
            elif np.all(F[idx2] <= F[idx1]) and np.any(F[idx2] < F[idx1]):
                return pop[idx2]
            else:
                return pop[idx1] if np.random.rand() < 0.5 else pop[idx2]

        parent1 = tournament_advanced(pop, F)
        parent2 = tournament_advanced(pop, F)

        # --- Apply Crossover ---
        if crossover is not None:
            offspring_array = np.array([parent1, parent2])
            # print("============> Crossover", offspring_array.shape)
            Y = crossover._do(problem, offspring_array)
            child1 = Y[0]
            child2 = Y[1]
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        # --- Apply Mutation ---
        if mutation is not None:
            offspring_matrix = np.array([child1, child2])
            # print("============> Mutation", offspring_array.shape)
            offspring_mut = mutation._do(problem, offspring_matrix)
            child1 = offspring_mut[0]
            child2 = offspring_mut[1]

        # --- Evaluate offspring ---
        offspring = [child1, child2]
        F_off = [problem.evaluate(c) for c in offspring]

        # --- Merge population and offspring ---
        pop += offspring
        F += F_off

        # # --- Survival: remove individual with smallest HV contribution ---
        # Keep pop_size best individuals based on hypervolume contribution
        while len(pop) > pop_size:

            contributions = []
            for i in range(len(pop)):
                temp_F = [f for j,f in enumerate(F) if j != i]
                hv_all = hv_2d(F, ref_point)
                hv_without = hv_2d(temp_F, ref_point)
                contributions.append(hv_all - hv_without)

            worst_idx = np.argmin(contributions)
            pop.pop(worst_idx)
            F.pop(worst_idx)

        if (gen+1) % 50 == 0:
            print(f"Generation {gen+1}/{n_gen} complete")

    return pop, F
