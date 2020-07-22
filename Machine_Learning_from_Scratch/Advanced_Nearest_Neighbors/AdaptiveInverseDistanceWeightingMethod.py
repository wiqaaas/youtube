
import numpy as np
from scipy.spatial import distance

class AIDW:
      def __init__(self, radius=None, closestNeighbors=None, simpleIDW=False, power=None):
            self.radius = radius
            self.closestNeighbors = closestNeighbors
            self.simpleIDW = simpleIDW
            self.power = power
            
      
      def searchNeighbor(self, testGrid, train_data, dist_metric = "euclidean"):
            radius = self.radius
            trainCoordinates = train_data[:,[0,1]]
            test_train_dst = distance.cdist(testGrid,trainCoordinates, dist_metric)  
            tmp = np.where(test_train_dst<=radius,True,False)         #Search Neigbourhood
            test_neighbors = []
            for i in range(0, testGrid.shape[0]):
                  neighbor_samples = train_data[tmp[i]]
                  neighbor_distance = test_train_dst[i][tmp[i]].reshape(-1,1)
                  neighbors = np.concatenate((neighbor_samples,neighbor_distance),axis=1)
                  test_neighbors.append(neighbors)
            return test_neighbors
      
      def closestNeighbor(self, testGrid, train_data, dist_metric = "euclidean"):
            number = self.closestNeighbors
            trainCoordinates = train_data[:,[0,1]]
            test_train_dst = distance.cdist(testGrid,trainCoordinates, dist_metric)  
            test_neighbors = []
            for i in range(0, testGrid.shape[0]):
                  indices = np.argsort(test_train_dst[i])
                  sampleNumbers = indices[list(range(number))]
                  neighbor_samples = train_data[sampleNumbers]
                  neighbor_distance = test_train_dst[i][sampleNumbers].reshape(-1,1)
                  neighbors = np.concatenate((neighbor_samples,neighbor_distance),axis=1)
                  test_neighbors.append(neighbors)
            return test_neighbors
      
      @staticmethod
      def RelativeClusteringValue(test_neighbors,train_coord):
            Robs_test = []
            for i in range(len(test_neighbors)):
                  Robs = np.mean(test_neighbors[i][:,3])
                  Robs_test.append(Robs)  
          
            samples = train_coord.shape[0]
            xmin = train_coord[:,0].min()
            xmax = train_coord[:,0].max()
            ymin = train_coord[:,1].min()
            ymax = train_coord[:,1].max()
            Area = (xmax-xmin)*(ymax-ymin)
            Rexp_train = 1/(2*np.sqrt(samples/Area))
            
            RelativeDispersionValue = Robs_test/Rexp_train
            
            Rmin = RelativeDispersionValue.min()
            Rmax = RelativeDispersionValue.max()
            R_norm = (RelativeDispersionValue-Rmin)/(Rmax-Rmin)
            
            return R_norm
      
      @staticmethod
      def func1(x,a,b):
            return (b-x)/(b-a)
      
      def func2(self,x,a,b):
            return 1-self.func1(x,a,b)
      
      def membership(self,x,ranges):
            weightDecayMem = np.eye(len(ranges))
            for i in range(len(ranges)):
                  if x==ranges[i]:
                        return ranges[i]
                  
            weightDecayMem = np.zeros(len(ranges))
            for i in range(1,len(ranges)):
                  a = ranges[i-1]
                  b = ranges[i]
                  if x > a and x < b:
                        weightDecayMem[i-1] = self.func1(x,a,b)
                        weightDecayMem[i] = self.func2(x,a,b)
                        return weightDecayMem
                  
      def predict(self,test_neighbors,power):
            predictions = []     
            for i in range(len(test_neighbors)):
                  grades = test_neighbors[i][:,2]
                  dist = test_neighbors[i][:,3]
                  if np.any(dist==0):            
                        predictions.append(grades[dist==0])
                        continue
                  if self.simpleIDW==True:
                        weights = (1/dist)**power
                  else:
                        weights = (1/dist)**power[i]
                  weightsSum = np.sum(weights)
                  finalWeights = weights/weightsSum
                  #print(np.sum(finalWeights))
                  predictions.append(np.sum(finalWeights*grades))
            return predictions
      
      def fit(self, train_data, testGrid, dist_metric="euclidean",
              weightDecayLst=None,rangesLst=None):
            
            test_neighbors = self.closestNeighbor(testGrid, train_data, dist_metric="euclidean")
            if self.simpleIDW==True:
                  predictions = self.predict(test_neighbors,self.power)    
            else:                  
                  train_coord = train_data[:,[0,1]]
                  relativeClustering = self.RelativeClusteringValue(test_neighbors,train_coord)               
                  dstWlst = []     
                  for i in relativeClustering:
                        memVal = self.membership(i,rangesLst)   
                        dstWeight = weightDecayLst*memVal
                        dstWlst.append(np.sum(dstWeight))
                  predictions = self.predict(test_neighbors,dstWlst)
                  
            return predictions

                  
                  