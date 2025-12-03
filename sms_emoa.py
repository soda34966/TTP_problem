from TTPProblem import TTPProblemAdapter
import numpy as np
import warnings
import os
import glob
import multiprocessing
import matplotlib.pyplot as plt

# --- Library Imports ---
from pymoo.algorithms.moo.sms import SMSEMOA
from purealgorithm import smsemoa
from pymoo.optimize import minimize
from pymoo.core.duplicate import NoDuplicateElimination

# --- Custom Imports ---
# from smsemoa import SMSEMOA
from sms_logic import run_parallel_ils, optimize_tour_orientation
from problem_logic import TTPProblem, TunableSpectrumSampling, SmartMutation, CustomCrossover

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

    # 4. Initialize Main Evolutionary Process
    problem = TTPProblem(selected_path, best_tour)
    
    print(">>> Sampling Initial Population (Locked Tour + Cloud)...")
    sampler = TunableSpectrumSampling()
    X_init = sampler._do(problem, 150)
    
    # Initial Eval for plotting
    out_init = {}
    problem._evaluate(X_init, out_init)
    F_init = out_init["F"]

    print(f">>> Initializing SMS-EMOA...")
    # algorithm = SMSEMOA(
    #     pop_size=100,
    #     sampling=X_init,
    #     crossover=CustomCrossover(problem),
    #     mutation=SmartMutation(problem),
    #     eliminate_duplicates=NoDuplicateElimination()
    # )

    # res = minimize(problem, algorithm, ('n_gen', 500), seed=42, verbose=True)
    adapter = TTPProblemAdapter(problem)

    final_pop, final_F = smsemoa(
        adapter, 
        n_var=problem.n_cities + problem.n_items, 
        n_obj=2, 
        pop_size=100, 
        n_gen=500,
        sampler=sampler,
        crossover=CustomCrossover(problem),
        mutation=SmartMutation(problem)
    )
    print("================================")
    print("Final objective values:")
    print(np.array(final_F))
    print("================================")

    # 5. Save and Plot Results
    f_path = os.path.join(res_f_dir, f"RES_{base_name}.f")
    final_F = np.array(final_F)
    sorted_indices = np.argsort(final_F[:, 0])
    F_sorted = final_F[sorted_indices]
    
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

    plt.scatter(F_init[:, 0], F_init[:, 1], c='gray', marker='x', s=40, alpha=0.4, label='Initial')
    plt.plot(F_sorted[:, 0], F_sorted[:, 1], 'r-o', linewidth=1.5, markersize=5, label='Optimized')
    plt.xlabel("Travel Time"); plt.ylabel("Negative Profit")
    plt.title(f"TTP Evolution: {base_name}")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(graph_path, dpi=150)
    print(f">>> Done. Graph saved to {graph_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()