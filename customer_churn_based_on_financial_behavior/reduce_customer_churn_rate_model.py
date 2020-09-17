# Import Libraries

import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt

dataset = pd.read_csv('new_churn_data.csv')

# Data Preparation
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])