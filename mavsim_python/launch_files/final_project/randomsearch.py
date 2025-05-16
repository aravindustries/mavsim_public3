import os
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import tempfile
from tqdm import tqdm

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

for i in tqdm(range(1000), desc='random searching...'):
    command = [
        "python3", "launch_files/chap06/test_k11.py",
        "--cki", str(np.random.uniform(low=0, high=40.0)),
        "--ckp", str(np.random.uniform(low=0, high=50.0)),
        "--rkp", str(np.random.uniform(low=0, high=3.0)),
        "--rkd", str(np.random.uniform(low=0, high=10.0)),
        "--yk", str(np.random.uniform(low=0.1, high=10.0)),
        "--yw", str(np.random.uniform(low=0.1, high=20.0)),
        "--alki", str(np.random.uniform(low=0.0, high=10.0)),
        "--alkp", str(np.random.uniform(low=0.0, high=5.0)),
        "--pkp", str(np.random.uniform(low=-10.0, high=20.0)),
        "--pkd", str(np.random.uniform(low=0.0, high=5.0)),
        "--askp", str(np.random.uniform(low=0.0, high=10.0)),
        "--aski", str(np.random.uniform(low=0.0, high=5.0)),
    ]

    subprocess.call(command, stderr=subprocess.DEVNULL)