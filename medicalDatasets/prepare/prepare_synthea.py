import pandas as pd
import numpy as np

dfA = pd.read_csv("blockingWorkflows/data/medical/rawdata/syntheaA.csv")
dfB = pd.read_csv("blockingWorkflows/data/medical/rawdata/syntheaB.csv")

# Copy 500 records from A into B with noise
np.random.seed(0)
sample = dfA.sample(500).copy()
gt_pairs = []

for _, row in sample.iterrows():
    noisy = row.copy()
    noisy['Id'] = 'DUP_' + str(row['Id'])  # new ID for B copy
    # Add noise to name
    if isinstance(noisy['FIRST'], str) and len(noisy['FIRST']) > 2:
        pos = np.random.randint(0, len(noisy['FIRST']))
        noisy['FIRST'] = noisy['FIRST'][:pos] + noisy['FIRST'][pos+1:]
    gt_pairs.append({'id1': row['Id'], 'id2': noisy['Id']})
    dfB = pd.concat([dfB, noisy.to_frame().T], ignore_index=True)

dfB.to_csv("blockingWorkflows/data/medical/rawdata/syntheaB_with_dups.csv", index=False)
pd.DataFrame(gt_pairs).to_csv("blockingWorkflows/data/medical/rawdata/synthea_groundtruth.csv", index=False)