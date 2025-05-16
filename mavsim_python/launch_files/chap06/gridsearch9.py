import os
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import tempfile
from tqdm import tqdm


for i in tqdm(range(100), desc='random searching...'):
    command = [
        "python3", "launch_files/chap06/test_k.py",
        "--askp", str(np.random.uniform(low=-50.0, high=50.0)),
        "--aski", str(np.random.uniform(low=-50.0, high=50.0)),
    ]
    subprocess.call(command)