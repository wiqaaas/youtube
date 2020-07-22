
import pandas as pd
from scipy.stats import pearsonr

train = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

train_data = train[['x','y','v']].values
x = train_data[:,0]
y = train_data[:,1]
grades = train_data[:,2]

from Kriging import Kriging

#Semivariogram calculation
Azimuth=0
Azi_Tolerance=180
Lag_Distance=10
Lag_Tolerance=5
metric="Euclidean"
       
#Variogram modeling after semivariogram Calculation
sill = 95000
nugget = 38000
maxRange = 30 #range of variogram model where maximum variance is achieved.
variogramType = 'spherical' #variogram type

#For prediction      
originX = 1
originY = 1  
cellsizeX = 1
cellsizeY =1
xMax = 260
yMax = 300
neighborhood_radius = 30

model = Kriging(x,y,grades,
                 originX,originY,cellsizeX,cellsizeY,xMax,yMax,
                 neighborhood_radius,
                 variogramType, sill, nugget, maxRange,
                 Azimuth,Azi_Tolerance,Lag_Distance,Lag_Tolerance,metric)

predictions = model.predict()

corr = pearsonr(test['v'], predictions)
print(corr)
