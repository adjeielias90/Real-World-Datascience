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
dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])
