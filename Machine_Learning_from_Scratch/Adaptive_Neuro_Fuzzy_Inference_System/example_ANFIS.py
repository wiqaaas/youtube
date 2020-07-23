
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

Data = data.loc[:,['x','y','v']]
test = test.loc[:,['x','y','v']]

#Data = data.loc[:,['FC','VM','Ash','Moisture','Cal V']]

# digitizing continuous variable
aa = Data['v']
minima = aa.min()
maxima = aa.max()
bins = np.linspace(minima-1,maxima+1, 3)
binned = np.digitize(aa, bins)
plt.hist(binned, bins=50)
data_train, data_test = train_test_split(Data, test_size=0.2,
                                          random_state=101,stratify=binned)

X_train = data_train.drop("v",axis=1).values
y_train = data_train["v"].copy().values
X_test = data_test.drop("v",axis=1).values
y_test = data_test["v"].copy().values
X_val = test.drop("v",axis=1).values
y_val = test["v"].copy().values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
#X_val = scaler_x.transform(X_val)
scaler_y.fit(y_train.reshape(-1,1))
y_train = scaler_y.transform(y_train.reshape(-1,1))
y_test = scaler_y.transform(y_test.reshape(-1,1))
#y_val = scaler_y.transform(y_val.reshape(-1,1))


from ANFIS import EVOLUTIONARY_ANFIS

E_Anfis = EVOLUTIONARY_ANFIS(functions=3,generations=500,offsprings=10,
                             mutationRate=0.2,learningRate=0.2,chance=0.7,ruleComb="simple")

bestParam, bestModel = E_Anfis.fit(X_train,y_train,optimize_test_data=False)

bestParam, bestModel = E_Anfis.fit(X_train,y_train,X_test,y_test,optimize_test_data=True)

from scipy.stats import pearsonr
pred_train = E_Anfis.predict(X_train,bestParam,bestModel)
pearsonr(y_train,pred_train.reshape(-1,1))

pred_test = E_Anfis.predict(X_test,bestParam,bestModel)
pearsonr(y_test,pred_test.reshape(-1,1))
        