import os
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import tempfile
from tqdm import tqdm

# Parameter ranges
cki_range = np.linspace(0, 50, 3)
ckp_range = np.linspace(10, 20, 3)
rkp_range = np.linspace(1.0, 2.0, 2)
rkd_range = np.linspace(0.01, 0.05, 2)
alki_range = np.linspace(0.05, 0.1, 2)
alkp_range = np.linspace(0.1, 0.2, 2)
pkp_range = np.linspace(-2.0, -1.5, 2)
pkd_range = np.linspace(0.01, 0.05, 2)
askp_range = np.linspace(0.1, 0.2, 2)
aski_range = np.linspace(0.05, 0.1, 2)

# Create all parameter combinations
param_grid = list(product(
    cki_range, ckp_range, rkp_range, rkd_range,
    alki_range, alkp_range, pkp_range, pkd_range,
    askp_range, aski_range
))

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

for cki in tqdm(cki_range, desc="Sweeping cki"):
    for ckp in ckp_range:
        for rkp in rkp_range:
            for rkd in rkd_range:
                for alki in alki_range:
                    for alkp in alkp_range:
                        for pkp in pkp_range:
                            for pkd in pkd_range:
                                command = [
                                    "python3", "launch_files/chap06/test_k.py",
                                    "--cki", str(cki),
                                    "--ckp", str(ckp),
                                    "--rkp", str(rkp),
                                    "--rkd", str(rkd),
                                    "--yk", str(default_yk),
                                    "--alki", str(alki),
                                    "--alkp", str(alkp),
                                    "--pkp", str(pkp),
                                    "--pkd", str(pkd),
                                    "--askp", str(default_askp),
                                    "--aski", str(default_aski),
                                ]

                                subprocess.call(command)