import pandas as pd

from Fcmeans_regression_global import REGRESSION 

train = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

data_train = train[['x','y','v']]
data_test = test[['x','y','v']]

X_train = data_train[['x','y']].values
y_train = data_train['v'].values
X_test = data_test[['x','y']].values
y_test = data_test['v'].values

sigma = 2
F_regression = REGRESSION(sigma,n_clusters=5, max_iter=150, fuzzines=1.5, error=1e-5, 
                          random_state=42, dist="euclidean", method="Cmeans",outputCov=True)

finalPred = F_regression.fit_regression(X_train,y_train,X_test)

from scipy.stats import pearsonr
pearsonr(finalPred,y_test)

