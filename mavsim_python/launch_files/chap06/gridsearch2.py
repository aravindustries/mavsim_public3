import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from datetime import datetime

# Default constants
default_cki = np.float64(20.267588066747148)
default_ckp = np.float64(14.37363238287152)
default_rkp = np.float64(1.4)
default_rkd = np.float64(0.02102713480353399)
default_yk = 2.0
default_yw = 4.0
default_alki = np.float64(0.07886358308644499)
default_alkp = np.float64(0.19471708332776536)
default_pkp = np.float64(-1.8257575757575761)
default_pkd = np.float64(0.03103419748228041)
default_askp = 0.1484412347207722
default_aski = 0.06853720758023289

# Create parameter ranges (you can adjust these)
cki_range = np.linspace(18.0, 22.0, 3)
ckp_range = np.linspace(13.0, 16.0, 3)
rkp_range = np.linspace(1.2, 1.6, 3)
rkd_range = np.linspace(0.015, 0.03, 3)
alki_range = np.linspace(0.06, 0.10, 3)
alkp_range = np.linspace(0.15, 0.25, 3)

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"simulation_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all parameter combinations
param_grid = list(itertools.product(cki_range, ckp_range, rkp_range, rkd_range, alki_range, alkp_range))

print(f"Running {len(param_grid)} simulations...")

best_score = float('inf')
best_params = None
best_filename = None

for i, (cki, ckp, rkp, rkd, alki, alkp) in enumerate(param_grid):
    filename = f"sim_{i:03d}_cki{cki:.2f}_ckp{ckp:.2f}_rkp{rkp:.2f}_rkd{rkd:.3f}_alki{alki:.3f}_alkp{alkp:.3f}.csv"
    output_path = os.path.join(output_dir, filename)
    
    command = [
        "python3", "launch_files/chap06/test_k.py",
        "--cki", str(cki),
        "--ckp", str(ckp),
        "--rkp", str(rkp),
        "--rkd", str(rkd),
        "--yk", str(default_yk),
        "--alki", str(alki),
        "--alkp", str(alkp),
        "--pkp", str(default_pkp),
        "--pkd", str(default_pkd),
        "--askp", str(default_askp),
        "--aski", str(default_aski),
        "--output", output_path
    ]

    print(f"[{i+1}/{len(param_grid)}] Running: {filename}")
    try:
        subprocess.run(command, check=True)
        
        # Read RMSE values from output CSV
        df = pd.read_csv(output_path)
        
        # Sum of all RMSEs (you can customize this)
        rmse_columns = [col for col in df.columns if 'rmse' in col.lower()]
        total_rmse = df[rmse_columns].iloc[0].sum()

        if total_rmse < best_score:
            best_score = total_rmse
            best_params = (cki, ckp, rkp, rkd, alki, alkp)
            best_filename = filename

    except subprocess.CalledProcessError:
        print(f"Simulation failed for: {filename}")
    except Exception as e:
        print(f"Error reading output for {filename}: {e}")

print("\nAll simulations complete.")
print(f"\n✅ Best parameter set: cki={best_params[0]}, ckp={best_params[1]}, rkp={best_params[2]}, "
      f"rkd={best_params[3]}, alki={best_params[4]}, alkp={best_params[5]}")
print(f"✅ Minimum total RMSE: {best_score}")
print(f"✅ Output file: {best_filename}")
