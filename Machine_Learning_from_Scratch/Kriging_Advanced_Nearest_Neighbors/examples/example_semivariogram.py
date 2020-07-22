
import pandas as pd

train = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

train_data = train[['x','y','v']].values
x = train_data[:,0]
y = train_data[:,1]
grades = train_data[:,2]

from Kriging import Variogram

Azimuth=0
Azi_Tolerance=180
Lag_Distance=10
Lag_Tolerance=5
metric="Euclidean"
                  
model = Variogram(x,y,grades,
                  Azimuth,Azi_Tolerance,Lag_Distance,Lag_Tolerance,
                  metric)

#SemiVariogram Calculation
semiVariance = model.Semivariogram()

#Plotting 
import matplotlib.pyplot as plt

lagDistanceLst = model.getLagDistLst()

fig = plt.figure(figsize=[12,4], dpi = 80)
plt.scatter(lagDistanceLst, semiVariance)
plt.xlabel('lag_distance')
plt.ylabel('semivariance')
plt.suptitle('experimental_variogram', fontsize=18,x=0.45,y=0.9)

#Interactive plotting
import plotly.express as px
from plotly.offline import plot
fig = px.scatter(x=lagDistanceLst, y=semiVariance)
plot(fig)
