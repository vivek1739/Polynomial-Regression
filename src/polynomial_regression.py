# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data and train/test data
processeddata_path = os.path.join(os.path.pardir,'data','processed')
X = pd.read_csv(os.path.join(processeddata_path,'X.csv')).values
y = pd.read_csv(os.path.join(processeddata_path,'y.csv')).values


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

########### for degree =2 
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

# visualizeing the linear regression
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()

############ for degree = 4 
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly[:,1:], y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly[:,1:],y)

# visualizeing the linear regression
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X_poly[:,1:]), color='blue')
plt.show()