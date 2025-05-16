import os
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile

default_yk = 2.0
rounds = 3               # Number of refinement rounds
samples_per_round = 200  # Number of samples per round
top_k = 10               # Number of top performers to refine around

# Initial parameter search space (min, max) for each parameter
search_space = {
    "cki": (0, 50),
    "ckp": (10, 20),
    "rkp": (1.0, 2.0),
    "rkd": (0.01, 0.05),
    "alki": (0.05, 0.1),
    "alkp": (0.1, 0.2),
    "pkp": (-2.0, -1.5),
    "pkd": (0.01, 0.05),
    "askp": (0.1, 0.2),
    "aski": (0.05, 0.1)
}

def run_sim(params):
    keys = list(search_space.keys())
    param_dict = dict(zip(keys, params))

    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.csv') as temp_file:
        result_path = temp_file.name

    command = [
        "python3", "launch_files/chap06/test_k.py",
        "--cki", str(param_dict["cki"]),
        "--ckp", str(param_dict["ckp"]),
        "--rkp", str(param_dict["rkp"]),
        "--rkd", str(param_dict["rkd"]),
        "--yk", str(default_yk),
        "--alki", str(param_dict["alki"]),
        "--alkp", str(param_dict["alkp"]),
        "--pkp", str(param_dict["pkp"]),
        "--pkd", str(param_dict["pkd"]),
        "--askp", str(param_dict["askp"]),
        "--aski", str(param_dict["aski"]),
        "--outfile", result_path
    ]

    subprocess.call(command)

    try:
        with open(result_path, 'r') as f:
            rmse = [float(line.strip()) for line in f]
    except Exception:
        rmse = [np.nan, np.nan, np.nan]

    os.remove(result_path)

    result = param_dict.copy()
    result.update({
        "airspeed": rmse[0],
        "altitude": rmse[1],
        "course": rmse[2]
    })
    return result

def sample_params(space):
    return [
        tuple(np.random.uniform(*space[key]) for key in space)
        for _ in range(samples_per_round)
    ]

def refine_search_space(top_df):
    refined = {}
    for key in search_space:
        low = top_df[key].min()
        high = top_df[key].max()
        margin = (high - low) * 0.5
        global_low, global_high = search_space[key]
        refined[key] = (
            max(global_low, low - margin),
            min(global_high, high + margin)
        )
    return refined

if __name__ == '__main__':
    overall_results = []

    for round_idx in range(1, rounds + 1):
        print(f"\n Round {round_idx} — Sampling from parameter space:")
        param_grid = sample_params(search_space)

        with Pool(processes=cpu_count()) as pool:
            round_results = list(tqdm(pool.imap(run_sim, param_grid), total=len(param_grid)))

        df = pd.DataFrame(round_results)
        df["score"] = df[["airspeed", "altitude", "course"]].mean(axis=1)

        overall_results.append(df)
        df.to_csv(f"random_search_round_{round_idx}.csv", index=False)

        
        # Drop any rows where score is NaN
        df = df.dropna(subset=["airspeed", "altitude", "course"])
        df["score"] = df[["airspeed", "altitude", "course"]].mean(axis=1)

        if len(df) < top_k:
            print(f"⚠️ Warning: Only {len(df)} valid simulations this round. Reducing top_k.")
            top_k_eff = len(df)
        else:
            top_k_eff = top_k

        top_df = df.nsmallest(top_k_eff, "score")

        search_space = refine_search_space(top_df)

    final_df = pd.concat(overall_results, ignore_index=True)
    final_df.to_csv("random_search_all_rounds.csv", index=False)

    print("\n Search complete. Top results:")
    print(final_df.nsmallest(5, "score"))
