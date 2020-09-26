# Begin Exploratory Data Analysis

# Imports
# Move all imports to beggining of file

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')

# Some housekeeping before EDA
# See how our data looks

dataset.head()
dataset.columns
dataset.describe()


# EDA
# Clean Data

# Removing NaN
dataset.isna().any() # No NaNs

# Plot Histogram

# Drop redundant tuples
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100

    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])