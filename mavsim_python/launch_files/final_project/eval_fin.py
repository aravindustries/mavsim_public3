import os 
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import tempfile
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def pick_best(csv_file):
    df = pd.read_csv(csv_file)

    scaler = MinMaxScaler()
    df['Va_RMSE'] = scaler.fit_transform(df[['Va_RMSE']])
    df['Alt_RMSE'] = scaler.fit_transform(df[['Alt_RMSE']])
    df['Chi_RMSE_deg'] = scaler.fit_transform(df[['Chi_RMSE_deg']])
    df['score'] = np.cbrt(df['Va_RMSE']) + np.cbrt(df['Alt_RMSE']) + np.cbrt(df['Chi_RMSE_deg'])

    print('parents')
    min_rows = df.nsmallest(1, 'score')
    print(min_rows)
    return min_rows


def eval_df(df, outfile):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        command = [
            "python3", "launch_files/chap06/testk3.py",
            "--cki", str(row['cki']),
            "--ckp", str(row['ckp']),
            "--rkp", str(row['rkp']),
            "--rkd", str(row['rkd']),
            "--yk", str(row['yk']),
            "--alki", str(row['alki']),
            "--alkp", str(row['alkp']),
            "--pkp", str(row['pkp']),
            "--pkd", str(row['pkd']),
            "--askp", str(row['askp']),
            "--aski", str(row['aski']),
            "--outfile", outfile,
        ]

        subprocess.call(command, stderr=subprocess.DEVNULL)


best_controller = pick_best('launch_files/chap06/gen49.csv')
eval_df(best_controller, 'launch_files/chap06/outfile.csv')