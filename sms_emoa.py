import numpy as np
import warnings
import os
import glob
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

# --- Custom Framework Imports ---
from emoa_framework import CustomSMSEMOA
from lts_logic import run_parallel_ils, optimize_tour_orientation
from problem_logic import TTPProblem, TunableSpectrumSampling, Mutation, Crossover

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION SECTION
# ==========================================
CONFIG = {
    # --- Pre-Processing ---
    'ils_time_limit': 240,       # Seconds for initial tour optimization
    
    # --- General GA Settings ---
    'pop_size': 100,            # Population Size
    'n_gen': 500,               # Number of Generations
    
    # --- Sampling (Initialization) ---
    'sample_sigma_min': 1.0,    # Min spread for Gaussian cloud
    'sample_sigma_max': 5.0,   # Max spread for Gaussian cloud (Exploration)
    
    # --- Mutation Probabilities ---
    'mut_pack_prob': 0.5,       # Prob. to mutate knapsack
    'mut_tour_prob': 0.3,       # Prob. to mutate tour
    
    # --- Specific Mutation Mechanics ---
    'mut_pack_flip_prob': 0.1,  # Chance to do a random bit flip
    'mut_pack_purge_prob': 0.15,# Chance to completely empty the knapsack
    'mut_tour_bridge_prob': 0.15# Chance to do a Double Bridge kick (Large jump)
}

# ==========================================
# EVALUATION LOGIC
# ==========================================
def get_non_dominated(F):
    n = F.shape[0]
    sorted_indices = np.argsort(F[:, 0])
    F_sorted = F[sorted_indices]
    
    is_dominated = np.zeros(n, dtype=bool)
    
    min_f2 = np.inf
    for i in range(n):
        if F_sorted[i, 1] >= min_f2:
            is_dominated[sorted_indices[i]] = True
        else:
            min_f2 = F_sorted[i, 1]
            
    return np.where(~is_dominated)[0]

def calculate_hypervolume(front, ref_point):
    valid_mask = (front[:, 0] <= ref_point[0]) & (front[:, 1] <= ref_point[1])
    front = front[valid_mask]
    
    if len(front) == 0: return 0.0

    front = front[np.argsort(front[:, 0])]
    
    hv = 0.0
    for i in range(len(front)):
        if i == len(front) - 1:
            width = ref_point[0] - front[i, 0]
        else:
            width = front[i+1, 0] - front[i, 0]
        
        height = ref_point[1] - front[i, 1]
        
        if width > 0 and height > 0:
            hv += width * height
    return hv

def compare_and_save(problem_name, my_pop, comp_dir, output_dir):
    print(f"\n{'='*60}\n>>> GECCO COMPETITION LEADERBOARD\n{'='*60}")
    os.makedirs(output_dir, exist_ok=True)
    
    raw_data_store = {}
    
    my_F = my_pop.get_F()
    my_F = my_F[np.argsort(my_F[:, 0])]
    raw_data_store["My_Algorithm"] = my_F

    search_dirs = [comp_dir, "result_compitition", "result_competition"]
    valid_comp_dir = None
    for d in search_dirs:
        if os.path.exists(d): valid_comp_dir = d; break
            
    if valid_comp_dir:
        clean_prob_name = problem_name.replace(".txt", "").strip()
        team_folders = sorted([f for f in os.listdir(valid_comp_dir) if os.path.isdir(os.path.join(valid_comp_dir, f))])
        
        for team in team_folders:
            team_path = os.path.join(valid_comp_dir, team)
            fname_1 = f"{team}_{clean_prob_name}.f"
            fname_2 = f"{team}_{clean_prob_name.replace('_', '-')}.f"
            
            target_file = None
            if os.path.isfile(os.path.join(team_path, fname_1)): target_file = os.path.join(team_path, fname_1)
            elif os.path.isfile(os.path.join(team_path, fname_2)): target_file = os.path.join(team_path, fname_2)
            
            if not target_file:
                patterns = [f"*{clean_prob_name}.f", f"*{clean_prob_name.replace('_', '-')}.f"]
                for pat in patterns:
                    matches = glob.glob(os.path.join(team_path, pat))
                    if matches: target_file = matches[0]; break

            if target_file:
                try:
                    valid_lines = []
                    with open(target_file, 'r') as f:
                        for line in f:
                            if re.match(r'^\s*[-+]?[\d\.]+', line):
                                try:
                                    vals = list(map(float, line.split()))
                                    if len(vals) >= 2: valid_lines.append(vals)
                                except: pass
                    
                    if not valid_lines: continue
                    data = np.array(valid_lines)
                    
                    if data.shape[1] >= 2:
                        time_col = data[:, -2]
                        profit_col = data[:, -1]
                        if np.mean(profit_col) > 0: neg_profit = -profit_col
                        else: neg_profit = profit_col 
                        raw_data_store[team] = np.column_stack((time_col, neg_profit))
                        print(f"   > Loaded {team:<15} | {len(time_col)} pts")
                except Exception as e: 
                    print(f"   ! Error {team}: {e}")

    if not raw_data_store: return

    all_solutions = np.vstack(list(raw_data_store.values()))
    nd_indices = get_non_dominated(all_solutions)
    nd_set = all_solutions[nd_indices]
    
    ideal_point = np.min(nd_set, axis=0)
    nadir_point = np.max(nd_set, axis=0)
    
    print(f"\n   > Reference Points (from Global ND Set):")
    print(f"     Ideal (Time, -Profit): {ideal_point}")
    print(f"     Nadir (Time, -Profit): {nadir_point}")
 
    ref_point_norm = np.array([1, 1]) 
    diff = nadir_point - ideal_point
    diff[diff == 0] = 1.0

    hv_results = {}
    report_path = os.path.join(output_dir, f"HV_Stats_{clean_prob_name}.txt")
    with open(report_path, 'w') as f_out:
        header = f"{'Rank':<5} {'Team':<20} {'Hypervolume':<10}"
        print("\n" + header); f_out.write(header + "\n")
        print("-" * 40); f_out.write("-" * 40 + "\n")

        for name, data in raw_data_store.items():
            local_nd_idx = get_non_dominated(data)
            data_nd = data[local_nd_idx]
            norm_data = (data_nd - ideal_point) / diff
            hv = calculate_hypervolume(norm_data, ref_point_norm)
            hv_results[name] = hv
        
        sorted_hv = sorted(hv_results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, hv) in enumerate(sorted_hv):
            line = f"{i+1:<5} {name:<20} {hv:.6f}"
            print(line); f_out.write(line + "\n")

    plt.figure(figsize=(10, 7))
    teams = list(raw_data_store.keys())
    colors = cm.get_cmap('tab20', len(teams))
    
    for i, team_name in enumerate(teams):
        data = raw_data_store[team_name]
        if team_name == "My_Algorithm":
            plt.scatter(data[:, 0], data[:, 1], label=team_name, color='red', marker='o', s=30, alpha=1.0, zorder=10)
        else:
            plt.scatter(data[:, 0], data[:, 1], label=team_name, facecolors='none', edgecolors=colors(i), marker='o', s=20, alpha=0.7, linewidth=1.2)

    plt.xlabel("time"); plt.ylabel("negative profit")
    time_pad = (nadir_point[0] - ideal_point[0]) * 0.1
    prof_pad = (nadir_point[1] - ideal_point[1]) * 0.1
    plt.xlim(ideal_point[0] - time_pad, nadir_point[0] + time_pad)
    plt.ylim(ideal_point[1] - prof_pad, nadir_point[1] + prof_pad)
    plt.legend(loc='upper right', frameon=True, edgecolor='gray', fancybox=True)
    plt.title(f"Pareto Fronts: {clean_prob_name}")
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    
    graph_path = os.path.join(output_dir, f"COMPARE_{clean_prob_name}.png")
    plt.savefig(graph_path, dpi=200)
    print(f"\n>>> Comparison graph saved to {graph_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    resource_dir = "TTP_resoure"
    res_f_dir = "result_f"
    res_x_dir = "result_x"
    res_graph_dir = "result_graph"
    final_compare_dir = "final_compare"
    comp_results_dir = "result_compitition" 
    
    os.makedirs(res_f_dir, exist_ok=True)
    os.makedirs(res_x_dir, exist_ok=True)
    os.makedirs(res_graph_dir, exist_ok=True)
    os.makedirs(final_compare_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(resource_dir, "*.txt")))
    if not files: print("No files found."); return

    print(f"\n Available Problems:")
    for i, f in enumerate(files): print(f" [{i}] {os.path.basename(f)}")
    try: idx = int(input("Select problem index: ")); selected_path = files[idx]
    except: return

    file_name = os.path.basename(selected_path)
    base_name = file_name.replace(".txt", "")

    # 1. ILS Optimization
    temp_nodes, temp_items = [], []
    with open(selected_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        node_sec, item_sec = False, False
        for line in lines:
            if "NODE_COORD_SECTION" in line: node_sec = True; item_sec=False; continue
            if "ITEMS SECTION" in line: item_sec = True; node_sec=False; continue
            parts = line.split()
            if not parts or not parts[0].isdigit(): continue
            if node_sec: temp_nodes.append([float(parts[1]), float(parts[2])])
            if item_sec: temp_items.append([float(parts[1]), float(parts[2]), int(parts[3])-1])

    nodes_arr = np.array(temp_nodes)
    items_arr = np.array(temp_items)
    
    print(f">>> Running ILS Pre-Optimization (Limit: {CONFIG['ils_time_limit']}s)...")
    best_tour = run_parallel_ils(nodes_arr, time_limit=CONFIG['ils_time_limit']) 
    best_tour = optimize_tour_orientation(nodes_arr, best_tour, items_arr)

    # 2. EVOLUTION
    problem = TTPProblem(selected_path, best_tour)
    print(f">>> Problem: Cap={problem.capacity}, v_max={problem.v_max}")
    print(f">>> Config: Pop={CONFIG['pop_size']}, Gen={CONFIG['n_gen']}")
    
    # Initialize components with Config
    sampling = TunableSpectrumSampling(
        sigma_min=CONFIG['sample_sigma_min'], 
        sigma_max=CONFIG['sample_sigma_max']
    )
    
    mutation = Mutation(problem, config=CONFIG)
    
    algorithm = CustomSMSEMOA(problem, CONFIG['pop_size'], sampling, Crossover(problem), mutation)
    final_pop = algorithm.run(n_gen=CONFIG['n_gen'], verbose=True)

    # 3. Save
    res_F = final_pop.get_F()
    res_X = final_pop.get_X()
    
    sorted_indices = np.argsort(res_F[:, 0])
    F_sorted = res_F[sorted_indices]
    X_sorted = res_X[sorted_indices]
    
    with open(os.path.join(res_f_dir, f"RES_{base_name}.f"), 'w') as f:
        for row in F_sorted: f.write(f"{row[0]:.2f} {-row[1]:.2f}\n")
    
    with open(os.path.join(res_x_dir, f"RES_{base_name}.x"), 'w') as f:
        for x_row in X_sorted:
            tour_keys = x_row[:problem.n_cities]
            tour = np.argsort(tour_keys) + 1 
            pack_genes = x_row[problem.n_cities:]
            packing = (pack_genes > 0.5).astype(int)
            f.write(f"{' '.join(map(str, tour))}\n")
            f.write(f"{' '.join(map(str, packing))}\n")
            f.write("\n")

    # 4. Compare
    compare_and_save(file_name, final_pop, comp_results_dir, final_compare_dir)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()