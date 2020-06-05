import pandas as pd
import numpy as np

results = './results/testLR_rep%i'

n_reps = 10

table = np.zeros((2,4))

for r in range(n_reps):
    result_file = (results + '/table.csv') % (r)

    data = pd.read_csv(result_file, dtype=float, usecols=[1,2,3,4])

    table += data.values

table /= n_reps

for col, name in enumerate(data.columns):
    data[name] = table[:, col]

data.to_csv((results + '/table_all_reps.csv') % 0)