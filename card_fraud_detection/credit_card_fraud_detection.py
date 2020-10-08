
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

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

X_train.shape
X_test.shape

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)