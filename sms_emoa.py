import numpy as np
import warnings
import os
import glob
import multiprocessing
import matplotlib.pyplot as plt

# --- Custom Framework Imports ---
from emoa_framework import CustomSMSEMOA
from sms_logic import run_parallel_ils, optimize_tour_orientation
from problem_logic import TTPProblem, TunableSpectrumSampling, Mutation, Crossover

warnings.filterwarnings("ignore")

def main():
    resource_dir = "TTP_resoure"
    hpi_dir = "HPI"
    res_f_dir = "result_f"
    res_graph_dir = "result_graph"
    
    os.makedirs(res_f_dir, exist_ok=True)
    os.makedirs(res_graph_dir, exist_ok=True)

    # 1. Select Problem
    files = sorted(glob.glob(os.path.join(resource_dir, "*.txt")))
    if not files: 
        print("No files found in TTP_resoure/ directory.")
        return

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
        return

    # 2. Temp Load Data for ILS (Core Optimization)
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
    ils_time = 180
    best_tour = run_parallel_ils(nodes_arr, time_limit=ils_time)
    best_tour = optimize_tour_orientation(nodes_arr, best_tour, items_arr)

    # 4. Initialize Main Evolutionary Process
    problem = TTPProblem(selected_path, best_tour)
    
    print(">>> Initializing Custom SMS-EMOA...")
    
    # Instantiate Custom Algorithm
    algorithm = CustomSMSEMOA(
        problem=problem,
        pop_size=20,
        sampling=TunableSpectrumSampling(),
        crossover=Crossover(problem),
        mutation=Mutation(problem)
    )

    print(f">>> Starting Optimization...")
    
    # Run Algorithm
    final_pop = algorithm.run(n_gen=200, verbose=True)

    # 5. Save and Plot Results
    res_F = final_pop.get_F()
    
    # Sort for plotting
    sorted_indices = np.argsort(res_F[:, 0])
    F_sorted = res_F[sorted_indices]
    
    f_path = os.path.join(res_f_dir, f"RES_{base_name}.f")
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
                if hpi[0, 1] > 0: hpi[:, 1] = -hpi[:, 1]
                hpi = hpi[np.argsort(hpi[:, 0])]
                plt.plot(hpi[:, 0], hpi[:, 1], 'g--', linewidth=2.5, alpha=0.9, label='HPI Reference')
        except: pass

    # --- FIX: Retrieve and Plot Initial Population ---
    F_init = algorithm.initial_F
    if F_init is not None:
        plt.scatter(F_init[:, 0], F_init[:, 1], c='gray', marker='x', s=40, alpha=0.4, label='Initial')

    # Plot final front
    plt.plot(F_sorted[:, 0], F_sorted[:, 1], 'r-o', linewidth=1.5, markersize=5, label='Optimized')
    plt.xlabel("Travel Time"); plt.ylabel("Negative Profit")
    plt.title(f"TTP Evolution: {base_name}")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(graph_path, dpi=150)
    print(f">>> Done. Graph saved to {graph_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()