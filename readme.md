# SMS-EMOA for the Traveling Thief Problem (TTP)

## Project Structure

* **`sms_emoa.py`**: The main entry point script that runs the optimization algorithm.
* **`problem_logic.py`**: Defines the TTP problem, including the objective functions (Travel Time vs. Total Profit) and evaluation logic.
* **`sms_logic.py`**: Contains the core logic for the SMS-EMOA, including hypervolume contribution calculations and selection mechanisms.
* **`emoa_framework.py`**: General framework utilities for the evolutionary algorithm.
* **`TTP_resource/`**: Contains the benchmark problem instances.
* **`result_compitition/`**: Stores reference results (Pareto fronts) from other algorithms for comparison.
* **`result_f/`** & **`result_x/`**: Stores the output of the algorithm:
    * `.f` files: Objective function values.
    * `.x` files: Decision variable solutions (tours and packing plans).
* **`final_compare/`**: Generated plots and statistical analysis comparing this implementation against the competition data (includes Hypervolume stats).

## How it Works

The Traveling Thief Problem combines two classic combinatorial problems:
1.  **Traveling Salesperson Problem (TSP):** Minimizing travel time.
2.  **Knapsack Problem (KP):** Maximizing the profit of items picked up.

All parameter can be adjusted in sms_emoa.py in CONFIG section


## Prerequisites

* Python 3.x
* Matplotlib (for generating comparison plots)
* NumPy (for matrix operations)
* SciPy (for cKDTree in Iterated Local Search )

```bash
pip install numpy matplotlib