# Import libraries

import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('appdata10.csv')

# Viewing the data
dataset.head(10)

# Distribution of numerical variables
dataset.describe()
