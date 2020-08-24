# import libraries
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization

# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Support vector machines
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


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


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count")
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
#sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer_all, fit_reg=False)


# Let's check the correlation between the variables
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)

# MODEL TRAINING (PROBLEM SOLUTION)

# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)
# X

y = df_cancer['target']
# y


# Split data into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

# Sings: I'm in love with the shape of you...
# get it? lmao nvm
X_train.shape
X_test.shape
y_train.shape
y_test.shape

svc_model = SVC()
svc_model.fit(X_train, y_train)