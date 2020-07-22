import numpy as np
import pandas as pd
from scipy.stats import pearsonr

train = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

train_coord = train[['x','y']].values
test_coord = test[['x','y']].values
train_data = train[['x','y','v']].values
test_data = test[['x','y','v']].values

from AdaptiveInverseDistanceWeightingMethod import AIDW

model = AIDW(closestNeighbors=5,simpleIDW=False, power=2)

bin0 = [0]
bin1 = [0.1]
bin2 = [0.4]
bin3 = [0.7]
bin4 = [0.8]
bin5 = [0.9]
bin6 = [1]
ranges = bin0+bin1+bin2+bin3+bin4+bin5+bin6

bin0 = [1]
bin1 = [1.5]
bin2 = [2]
bin3 = [2.5]
bin4 = [3]
bin5 = [10]
bin6 = [20]
weightDecay = bin0+bin1+bin2+bin3+bin4+bin5+bin6

predictions = model.fit(train_data,test_coord,dist_metric="euclidean",
                        weightDecayLst=weightDecay,rangesLst=ranges)

pred = np.array(predictions)
test = np.array(test_data[:,2])
pearsonr(test,pred)

val = test-pred
import matplotlib.pyplot as plt
plt.scatter(test_coord[:,0],test_coord[:,1],c=val,cmap="jet",vmin=val.min(),vmax=val.max())
plt.colorbar()