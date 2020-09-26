# Begin Exploratory Data Analysis

# Imports
# Move all imports to begining of file

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

# Plot Histograms

# Drop/Discard redundant tuples
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())

    # We have a big range for continous values. We don't want to take too much time to plot
    # when we have more than a 100 unique values, so we will set the number of values to 100 in such a case
    if vals >= 100:
        vals = 100

    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# Correlation with Response Variable (Note: Models like RF, SVMs are non-linear unlike this one)
# Will give us correlation of our dataset with our e-signed response variable

dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)


# Correlation Matrix to tell us the relationships between our columns(attributes)

## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
# This time we will use corr(), a more robust alternative to the corrwith() function
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
