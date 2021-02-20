## this is my final project for Analyzing Data with Python course taught by IBM and edX online

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge

# importing data, creating data frame, and creating a groupby of wine servings by continent
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/edx/project/drinks.csv')
display(df.head())
print(df.dtypes)
wine = df.groupby('continent')['wine_servings'].sum()
display(wine)

#creating a second dataframe of beer servings and continents, and performing statistical summary
df1 = df[['beer_servings', 'continent']]
display(df.groupby('continent')['beer_servings'].describe())

#displaying boxplot to show beer servings per country
ax1 = sns.boxplot(x = 'continent', y = 'beer_servings', data=df1)
plt.savefig('boxplot')
plt.figure()

#creating dataframe for wine servings and beer servings by continent
df2 = df[['beer_servings', 'wine_servings', 'continent']]

#using regplot for beer servings and wine servings
ax2 = sns.regplot(x = 'beer_servings', y = 'wine_servings', data=df2)
plt.savefig('regplot')
plt.figure()

#creating a linear regression model to predict total liters of pure alcohol using the number of wine servings
lr = LinearRegression()
x1 = df[['wine_servings']]
y1 = df[['total_litres_of_pure_alcohol']]
lr.fit(x1, y1)
yhat_total_litres = lr.predict(x1)
ax3 = sns.regplot(x1, y1).set_title('Total Litres of Alcohol as a Function of Servings of Wine')
plt.savefig('regplot of wine vs total alc consumed')
plt.figure()
print('The intercept is:', lr.intercept_)
print('The pearson coefficient is:', lr.coef_)
print('The R squared value is:', lr.score(x1, y1))

#performing multiple linear regression to predict total liters of alc consumption
lr1 = LinearRegression()
z = df[['beer_servings', 'spirit_servings', 'wine_servings']]
z_train, z_test, b_train, b_test = train_test_split(z, y1, test_size = 0.10, random_state = 0)
print(z_train.shape, b_train.shape)
print(z_test.shape, b_test.shape)
lr1.fit(z_train, b_train)
yhat_z = lr1.predict(z_test)
ax4 = sns.distplot(y1, hist=False, color='r', label='Actual Values')
sns.distplot(yhat_z, hist=False, color='b', label='Predicted Values', ax=ax4)
plt.title('Actual vs Predicted Total Litres of Pure Alcohol')
plt.savefig('distplot of total alc vs all other forms of alc')
plt.figure()
print('The R^2 values is:', lr1.score(z_test, b_test))

#creating pipeline to perform polynomial transformation on data set, and fit new linear regression model
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(z, y1)
ypipe = pipe.predict(z)
ax5 = sns.distplot(y1, hist=False, color = 'r', label='Actual Values')
sns.distplot(ypipe, hist=False, color='b', label='Predicted Values', ax=ax5)
plt.title('Actual vs Predicted Values')
plt.savefig('poly plot')
plt.figure()
print('The R^2 value for the polynomial model is:', pipe.score(z, y1))
# R^2 value is 0.887535 for the polynomial model

#creating and fitting a ridge regression object
RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(z_test, b_test)
yhat_ridge = RidgeModel.predict(z_test)
ax6 = sns.distplot(y1, hist=False, color = 'r', label='Actual Values')
sns.distplot(yhat_ridge, hist=False, color='b', label='Predicted Values', ax=ax6)
plt.title('Actual vs Predicted for Linear Ridge Model')
plt.savefig('linear ridge model')
plt.figure()
print('The R^2 value for the ridge model is:', RidgeModel.score(z_test, b_test))

# performing a 2nd order polynomial transformation on training and testing data, then creating and fitting a new ridge model
pr=PolynomialFeatures(degree=2)
z_train_pr = pr.fit_transform(z_train)
z_test_pr = pr.fit_transform(z_test)
b_train_pr = pr.fit_transform(b_train)
b_test_pr = pr.fit_transform(b_test)
RidgeModel2 = Ridge(alpha = 0.1)
RidgeModel2.fit(z_train_pr, b_train)
yhat_ridge_poly = RidgeModel2.predict(z_train_pr)
ax7 = sns.distplot(y1, hist=False, color = 'r', label='Actual Values')
sns.distplot(yhat_ridge_poly, hist=False, color='b', label='Predicted Values', ax=ax7)
plt.title('Actual vs Predicted for the Polynomial Ridge Model')
print('The R^2 value for the polynomial Ridge model is:', RidgeModel2.score(z_test_pr, b_test))
plt.savefig('poly ridge model')
plt.figure()









