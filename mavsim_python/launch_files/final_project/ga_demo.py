import os 
import subprocess
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import tempfile
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def pick_parents(n, csv_file):
    df = pd.read_csv(csv_file)

    scaler = MinMaxScaler()
    df['Va_RMSE'] = scaler.fit_transform(df[['Va_RMSE']])
    df['Alt_RMSE'] = scaler.fit_transform(df[['Alt_RMSE']])
    df['Chi_RMSE_deg'] = scaler.fit_transform(df[['Chi_RMSE_deg']])
    #print(df.head)

    df['score'] = np.cbrt(df['Va_RMSE']) + np.cbrt(df['Alt_RMSE']) + np.cbrt(df['Chi_RMSE_deg'])

    print('parents')
    min_rows = df.nsmallest(n, 'score')
    print(min_rows)
    return min_rows
    


def get_children_df(parent_df, num_children):
    children = []
    population = len(parent_df)
    for _ in range(num_children):
        # Randomly select two different parents
        idx1 = np.random.randint(0, population)
        idx2 = np.random.randint(0, population)
        while idx2 == idx1:
            idx2 = np.random.randint(0, population)

        P1 = parent_df.iloc[idx1]
        P2 = parent_df.iloc[idx2]

        # Arithmetic crossover
        alpha = np.random.uniform(0, 1)
        child = alpha * P1 + (1 - alpha) * P2

        # Mutation
        mutation_rate = 0.3
        for gain in ['cki', 'ckp', 'rkp', 'rkd', 'yk', 'yw', 'alki', 'alkp', 'pkp', 'pkd', 'askp', 'aski']:
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, 2)
                child[gain] *= noise
                #child[gain] = max(0, child[gain])  # Enforce non-negative gains

        # Add to children list
        children.append(child)

    # Combine all into a DataFrame
    children_df = pd.DataFrame(children).reset_index(drop=True)
    return children_df
        
        
def eval_df(df, outfile):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        command = [
            "python3", "launch_files/chap06/test_k2.py",
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


parent_df = pick_parents(10, 'launch_files/chap06/rmse_results7.csv')
child_df = get_children_df(parent_df, 16)
elite_df = pick_parents(4, 'launch_files/chap06/rmse_results7.csv')
population = pd.concat([elite_df, child_df], ignore_index=True)
print(population.head)
eval_df(population, 'launch_files/chap06/gen_demo0.csv')

for gen in range(1, 5):
    parent_df = pick_parents(10, 'launch_files/chap06/gen_demo'+str(gen-1)+'.csv')
    child_df = get_children_df(parent_df, 16)
    elite_df = pick_parents(4, 'launch_files/chap06/gen_demo'+str(gen-1)+'.csv')
    population = pd.concat([elite_df, child_df], ignore_index=True)
    print(population.head)
    eval_df(population, 'launch_files/chap06/gen_demo' + str(gen) + '.csv')