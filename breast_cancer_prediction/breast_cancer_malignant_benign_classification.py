# import libraries
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization

# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


cancer
# What dictionaries[keys] are present in our dataset?
cancer.keys()

# Print select tuples
print(cancer['DESCR'])
print(cancer['target_names'])
print(cancer['target'])


print(cancer['feature_names'])
print(cancer['data'])

# How is our data frame shaped?
cancer['data'].shape

# Pandas to the rescue!
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()
df_cancer.tail()