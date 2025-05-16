import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('launch_files/chap06/rmse_results2.csv')
#print(df.head)

scaler = MinMaxScaler()

df['Va_RMSE'] = scaler.fit_transform(df[['Va_RMSE']])
df['Alt_RMSE'] = scaler.fit_transform(df[['Alt_RMSE']])
df['Chi_RMSE_deg'] = scaler.fit_transform(df[['Chi_RMSE_deg']])
#print(df.head)

df['score'] = np.cbrt(df['Va_RMSE']) + np.cbrt(df['Alt_RMSE']) + np.cbrt(df['Chi_RMSE_deg'])
#print(df.head)

#pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.width', None)        # Disable line wrapping
print('min rows')
min_rows = df.nsmallest(5, 'score')
print(min_rows)

vectors = min_rows.values
cos_sim_matrix = cosine_similarity(vectors)
#print(cos_sim_matrix)

max_rows = df.nlargest(3, 'score')
#print(max_rows)
