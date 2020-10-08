
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler

np.random.seed(2)

# small EDA, visualize data

data = pd.read_csv('creditcard.csv')
data.head()

# Pre-processing

# let's quickly reshape our amount column
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

# remove what we don't need
data = data.drop(['Amount'],axis=1)

# see what we have
data.head()


data = data.drop(['Time'],axis=1)
data.head()