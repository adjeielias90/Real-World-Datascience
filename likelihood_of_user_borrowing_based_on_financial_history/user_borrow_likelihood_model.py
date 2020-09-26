# Begin Build Model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time

random.seed(100)


# Data Preprocessing
dataset = pd.read_csv('financial_data.csv')

# Feature Engineering

dataset = dataset.drop(columns = ['months_employed'])
# Better sanitize instances of period into months/years
dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])

# One Hot Encoding

dataset = pd.get_dummies(dataset)
dataset.columns
# Remove weird instances of period, make data linear
dataset = dataset.drop(columns = ['pay_schedule_semi-monthly'])


# Removing extra columns

# These columns are useful but will not be used in training
response = dataset["e_signed"]
# Save users into special variables
users = dataset['entry_id']
dataset = dataset.drop(columns = ["e_signed", "entry_id"])


# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    response,
                                                    test_size = 0.2,
                                                    random_state = 0)