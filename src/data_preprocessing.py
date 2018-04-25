# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data
rawdata_path = os.path.join(os.path.pardir,'data','raw')
processeddata_path = os.path.join(os.path.pardir,'data','processed')
dataset = pd.read_csv(os.path.join(rawdata_path,'Position_Salaries.csv'))

# getting X and y
X = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

# lets look at the scatter plot
plt.scatter(X,y,color='red')
plt.show()


# Splitting the data randomly 
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"""

# Save the processed data in data/processed
np.savetxt(os.path.join(processeddata_path,'X.csv'), X, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'y.csv'), y, delimiter=",")

