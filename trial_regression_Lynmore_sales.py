import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#From https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# load dataset
url = "/Users/Llewelyn_home/Dropbox/Computation_in_Python/AI/ResidentialSales Lynmore.xls"
cols = [7,8,9,11,15,16,17,19]
df = pd.read_excel(url,
                                sheetname=0,#ResidentialSales Lynmore
                                header=0,
                                parse_cols=cols)
#Missing data are just really hard to handle.
df.iloc[:,2:4]=df.iloc[:,2:4].astype(int,copy=False)/1e11
#Filling the missing list price values is difficult.
#a conservative approach is to assume that missing list prices are  the proportion
#of mean list to sale prices in order to introduce no more error than needed.
#The impact of this is to strongly coerce the data, however, to linearity.
df['List Price'].fillna(df['List Price'].mean()/df['Sale Price'].mean()*df['Sale Price'],inplace=True)
df.fillna(df.mean(),inplace=True)
dataset = df.values
# split into input (X) and output (Y) variables
X = dataset[:,[0,2,3,4,5,6,7]]
Y = dataset[:,1]

# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(28, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(14,kernel_initializer='normal',activation='relu'))
	model.add(Dense(7,kernel_initializer='normal',activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
seed = 7
np.random.seed(seed)
estimators = []
#estimators.append(('standardize', StandardScaler())) #messes with the final values
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=60, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: {0:.2f} ({1:.2f}) MSE".format(np.sqrt(results.mean()), results.std()))
print('Fitting model....')
pipeline.fit(X,Y)
pred_values= pipeline.predict(X)
print(pred_values[0:6])
print(Y[0:6])
#Using OLS model
import statsmodels.api as sm
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
#Using SKlearn
from sklearn import linear_model
lm = linear_model.LinearRegression()
model2 = lm.fit(X,Y)
SK_preds = lm.predict(X)
print(lm.coef_, lm.intercept_)
print(results.params)
#For pymc3 variant do....
#http://www.databozo.com/deep-in-the-weeds-complex-hierarchical-models-in-pymc3
#Try a Bayesian approach
import pymc3 as pm

with pm.Model() as mdl_ols:
    b0 = pm.Normal('b0', mu=0, sd=100)
    b1 = pm.Normal('b1', mu=0, sd=100)
    b2 = pm.Normal('b2', mu=0, sd=100)
    b3 = pm.Normal('b3', mu=0, sd=100)
    b4 = pm.Normal('b4', mu=0, sd=100)
    b5 = pm.Normal('b5', mu=0, sd=100)
    b6 = pm.Normal('b6', mu=0, sd=100)
    b7 = pm.Normal('b7', mu=0, sd=100)
    yest = b0 + b1 * X[:,0] + b2*X[:,1] +b3*X[:,2] +b4*X[:,3]+b5*X[:,4]+b6*X[:,5]+b7*X[:,6]
    sigma_y = pm.HalfCauchy('sigma_y', beta=10)
    likelihood = pm.Normal('likelihood', mu=yest, sd=sigma_y, observed=Y)
    traces_ols = pm.sample(10000,step=pm.Metropolis())
print(pm.summary(traces_ols))
pm.traceplot(traces_ols)
fig,ax = plt.subplots(1,1,figsize=(10,8))
ax.set_facecolor(plt.cm.gray(.95))
ax.set_title('Comparison of prediction methods for Lynmore housing.')
ax.set_xlabel('List price')
ax.set_ylabel('Sale price and predicted sale price')
ax.grid(True)
ax.plot(df['List Price'],pred_values,'b+', markersize=3,alpha=.8,label='NN model')
ax.plot(df['List Price'],df['Sale Price'],'k.',markersize=3,alpha=.8,label='Original data')
ax.plot(df['List Price'],results.predict(),'g|',markersize=5,alpha=.8,label='OLS model')
ax.plot(df['List Price'],SK_preds,'r.',markersize=2,alpha=.8,label='SKLearn')
ax.legend()
plt.show()
