
import numpy as np
from math import ceil
from scipy.spatial import distance

class Variogram(object):
    def __init__(self,x,y,grade,
                 Azimuth,Azi_Tolerance,Lag_Distance,Lag_Tolerance,
                 metric):
        
        self.x = x
        self.y = y
        self.grade = grade
        self.Azimuth = Azimuth 
        self.Azi_Tolerance = Azi_Tolerance
        self.Lag_Distance = Lag_Distance
        self.Lag_Tolerance = Lag_Tolerance
        self.metric = metric
        
    def pairwiseDiff(self,):
        """
        ===============[Finding v_pairwise difference, deltax and deltay matrices]==============
        """
        x = self.x
        y = self.y
        grades = self.grade
        N = x.shape[0]
        gradeChange = np.zeros((N,N))
        xChange = np.zeros((N,N))
        yChange = np.zeros((N,N))
        for i in range(N):
            gradeChange[i] = (grades[i]-grades)**2
            xChange[i] = (x[i]-x)
            yChange[i] = (y[i]-y)
        d_pairwiseDiff = {"gradeChange":gradeChange,"xChange":xChange,"yChange":yChange}
        return d_pairwiseDiff
        
    @staticmethod
    def skip_diag_masking(arr):
        """
        Removing diagnol elements
        """
        return arr[~np.eye(arr.shape[0],dtype=bool)].reshape(arr.shape[0],-1)
       
    def AzimuthArr(self):
        '''
        ####Azimuth###
        azimuth is angle with north in anticlockwise direction
        Returns azimuth angle
            formula :
            1) (90 - arctan(dy/dx)    when dx is positive (i.e. quadrant 1 and 4)
            Note for quadrant 4, arctan(dy/dx) is negative
            270 - arctan(dy/dx)    when dx is negative (i.e. quadrant 2 and 3)
            Note for quadrant 3, arctan(dy/dx) is negative)        
        '''    

        d_pairwiseDiff = self.pairwiseDiff()
        gradeChange,xChange,yChange = d_pairwiseDiff["gradeChange"],d_pairwiseDiff["xChange"],d_pairwiseDiff["yChange"]

        xChange_n = self.skip_diag_masking(xChange)
        yChange_n = self.skip_diag_masking(yChange)
        
        y_x_change = (yChange_n/xChange_n)
        angle_quad_1_4 = np.degrees((np.pi*0.5)-np.arctan(y_x_change))
        #np.min(angle_quad_1_4) is 0
        #np.max(angle_quad_1_4) is 180
        angle_quad_2_3 = np.degrees((np.pi*1.5)-np.arctan(y_x_change))
        #np.min(angle_quad_2_3) is 180
        #np.max(angle_quad_2_3) is 360
        arr_Azimuth = np.where(xChange_n>=0,angle_quad_1_4,angle_quad_2_3)
        return arr_Azimuth, gradeChange
    
    @staticmethod
    def Distance(x_train,metric):
        dst = ((distance.cdist(x_train,x_train, metric)))
        return dst


    def Semivariogram(self,):
    
        """
        for omnidirectional azimuth is 0 degrees and azimuth tolerance is 180 degrees,
        otherwise consider azimuth is 45 degrees and azimuth tolerance is 22.5 degrees,
        
        lagdistance is distance between two pairs and is usually equal to shortest sample spacing
        lagtorerance is half of lag distance
        """
        
        print("Semivariogram Calculation Started")
        x = self.x
        y = self.y 
        Azimuth = self.Azimuth 
        Azi_Tolerance = self.Azi_Tolerance 
        Lag_Distance = self.Lag_Distance 
        Lag_Tolerance = self.Lag_Tolerance 
        metric = self.metric
        
        coordinates = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        dst = self.Distance(coordinates, metric)
        dst_n = self.skip_diag_masking(dst)
        
        arr_Azimuth, gradeChange = self.AzimuthArr()
        
        gradeChange_n = self.skip_diag_masking(gradeChange)
        
        max_dst = np.max(dst_n)
        
        nLags = np.arange(0,ceil(max_dst/Lag_Distance))

        lagDistanceLst = nLags*Lag_Distance
        
        lag_range_min = lagDistanceLst-Lag_Tolerance
        lag_range_max = lagDistanceLst+Lag_Tolerance
        azimuth_range_min = Azimuth-Azi_Tolerance
        azimuth_range_max = Azimuth+Azi_Tolerance
        
        semiVariance = []
        for i in range(len(nLags)-1):
            """
            1) get pairs satisfying conditions for lag and azimuth conditions 
            2) count pairs
            3) sum change in grades for those pairs
            4) find variance by dividing sum change in grades with count pairs
            5) find semivariance by multiplying with 0.5.
            """
            pairs1 = np.where(dst_n<=lag_range_max[i],1,0)
            pairs2 = np.where(dst_n>lag_range_min[i],1,0)
            pairs3 = np.where(arr_Azimuth<=azimuth_range_max,1,0)
            pairs4 = np.where(arr_Azimuth>azimuth_range_min,1,0)
            pairs = pairs1*pairs2*pairs3*pairs4
            pairsCount = np.sum(pairs)
            gradeSum = np.sum(gradeChange_n[pairs==1])
            regionVariance = gradeSum/pairsCount
            regionSemiVariance = regionVariance*0.5
            semiVariance.append(regionSemiVariance)
            
        print("Semivariogram Calculation Ended")
                
        return semiVariance
    
    def getLagDistLst(self,):
        x = self.x
        y = self.y 
        metric = self.metric
        Lag_Distance = self.Lag_Distance 
        coordinates = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        dst = self.Distance(coordinates, metric)
        dst_n = self.skip_diag_masking(dst)
    
        max_dst = np.max(dst_n)
        
        nLags = np.arange(0,ceil(max_dst/Lag_Distance))
        lagDistance_Lst = nLags[:-1]*Lag_Distance

        return lagDistance_Lst
        
    def AllLags(self):
        lagDistanceLst = self.getLagDistLst() 
        maxLag = lagDistanceLst.max()
        allLags = np.arange(0,maxLag,1) #
        return allLags  
    
    def VariogramModel(self,variogramType, sill, nugget, maxRange, Data):

        sillContribution = sill-nugget
        
    #    modelVariance =nugget + (np.where(variogramType == 'sph', np.where(sillCont>0, np.where(allLags<maxRange,(C*(1.5*(allLags/maxRange)-0.5*(allLags/maxRange)**3)),C),np.empty([])),np.where(var == 'gaus', C*(1-np.exp(-3*(azm**2/Rmax**2))), C*(1-np.exp(-3*azm/Rmax)))))
        if variogramType == 'spherical':
            modelVariance = np.where(Data<=maxRange,nugget+sillContribution*(1.5*(Data/maxRange)-0.5*(Data/maxRange)**3),sill)
        elif variogramType == 'gaussian':
            modelVariance = nugget+ sillContribution*(1-np.exp(-3*(Data**2/maxRange**2)))
        elif  variogramType == 'exponential':
            modelVariance = nugget+ sillContribution*(1-np.exp(-3*(Data/maxRange)))
        return modelVariance
    
    def fitVariogram(self, variogramType, sill, nugget, maxRange):
        print("Variogram Fitting Started!")
        allLags = self.AllLags()
        model_Variance = self.VariogramModel(variogramType, sill, nugget, maxRange,allLags)
        print("Variogram Fitting Done!")
        return model_Variance
    
    
class Kriging(Variogram):
    
    def __init__(self,x,y,grade,
                 originX,originY,cellsizeX,cellsizeY,xMax,yMax,
                 neighborhood_radius,
                 variogramType, sill, nugget, maxRange,
                 Azimuth=0,Azi_Tolerance=180,Lag_Distance=10,Lag_Tolerance=5,metric="Euclidean"):
        
        Variogram.__init__(self,x,y,grade,
                 Azimuth,Azi_Tolerance,Lag_Distance,Lag_Tolerance,
                 metric)
        self.originX = originX
        self.originY = originY
        self.cellsizeX = cellsizeX
        self.cellsizeY = cellsizeY
        self.xMax = xMax
        self.yMax = yMax
        self.x = x
        self.y = y
        self.grade = grade
        self.radius = neighborhood_radius
        self.variogramType = variogramType 
        self.sill = sill 
        self.nugget = nugget 
        self.maxRange = maxRange

    @staticmethod
    def Grid(originX, originY, cellsizeX,cellsizeY, xMax, yMax):
        '''
        Define a grid for the estimation (Kriging portion)
            1) originX and originY are origin coordinates
            2) cellciseX and cellsizeY defines the cellsize/steps in x and y directions
            3) xMax and yMax is the maximum extent of grid in x and y directions
        ''' 
        x = np.arange(originX,xMax+1,cellsizeX)
        y = np.arange(originY,yMax+1,cellsizeY)
        xAxis, yAxis = np.meshgrid(x, y, sparse=True)
        grid = np.concatenate((xAxis,np.repeat(yAxis[0],xAxis.shape[1]).reshape(1,-1)),axis=0).T
        for i in range(1,yAxis.shape[0]):    
            arr0 = np.concatenate((xAxis,np.repeat(yAxis[i],xAxis.shape[1]).reshape(1,-1)),axis=0).T
            grid = np.concatenate((grid,arr0),axis=0)
        return grid

    def searchNeighbor(self):
        
        trainCoordinates = np.concatenate((self.x.reshape(-1,1),self.y.reshape(-1,1)),axis=1)
        train_data = np.concatenate((self.x.reshape(-1,1),self.y.reshape(-1,1),self.grade.reshape(-1,1)),axis=1)
        
        originX = self.originX
        originY = self.originY
        cellsizeX = self.cellsizeX
        cellsizeY = self.cellsizeY
        xMax = self.xMax
        yMax = self.yMax
        testGrid = self.Grid(originX, originY, cellsizeX,cellsizeY, xMax, yMax)
        
        radius = self.radius
        test_train_dst = distance.cdist(testGrid,trainCoordinates, "euclidean")   
        tmp = np.where(test_train_dst<=radius,True,False)         #Search Neigbourhood
        test_neighbors = []
        for i in range(0, testGrid.shape[0]):
            neigbor_samples = train_data[tmp[i]]
            test_neighbors.append(neigbor_samples)
        return test_neighbors, testGrid

    # Distance Matrix Calculation
    def distanceMatrix(self,):
        
        test_neighbors, testGrid = self.searchNeighbor()
        
        NeighborsPairwiseDstLst = []
        NeighborsTestDstLst = []
        for i in range(len(test_neighbors)):
            NeighborsCoordinates = test_neighbors[i][:,[0,1]]
            testCoordinates = [testGrid[i]]
            NeighborsPairwiseDst = distance.cdist(NeighborsCoordinates,NeighborsCoordinates, "euclidean")
            NeighborsTestDst = distance.cdist(testCoordinates,NeighborsCoordinates, "euclidean")
            NeighborsPairwiseDstLst.append(NeighborsPairwiseDst)
            NeighborsTestDstLst.append(NeighborsTestDst)
        return NeighborsPairwiseDstLst,NeighborsTestDstLst, test_neighbors

    # C_Matrix Calculation
    def CovarianceMatrices(self,):
        NeighborsPairwiseDstLst, NeighborsTestDstLst, test_neighbors = self.distanceMatrix()
        
        variogramType = self.variogramType
        sill = self.sill
        nugget = self.nugget
        maxRange = self.maxRange        
        testSamples = len(NeighborsPairwiseDstLst)
        
        Neighbors_pairwise_Covariance = []
        for i in range(testSamples):
            CovMatrix = self.VariogramModel(variogramType, sill, nugget, maxRange, NeighborsPairwiseDstLst[i])
            #ADDING LAGRANGE MULTIPLIER
            ones = np.ones(CovMatrix.shape[0])
            onesC = ones.reshape(-1,1)
            CovMatrix = np.concatenate((CovMatrix,onesC),axis=1)
            ones = np.ones(CovMatrix.shape[1])
            onesR = ones.reshape(1,-1)
            CovMatrix = np.concatenate((CovMatrix,onesR),axis=0)
            CovMatrix[-1,-1]=0           
            Neighbors_pairwise_Covariance.append(CovMatrix)            
        test_neighors_Covariance = []
        for i in range(testSamples):
            CovMatrix = self.VariogramModel(variogramType, sill, nugget, maxRange, NeighborsTestDstLst[i])
            #ADDING LAGRANGE MULTIPLIER
            ones = np.ones(CovMatrix.shape[0])
            onesC = ones.reshape(1,1)
            CovMatrix = np.concatenate((CovMatrix,onesC),axis=1)
            test_neighors_Covariance.append(CovMatrix)
    
        return Neighbors_pairwise_Covariance, test_neighors_Covariance, test_neighbors

    def predict(self):   
        Neighbors_pairwise_Covariance, test_neighors_Covariance, test_neighbors = self.CovarianceMatrices()
        testSamples = len(Neighbors_pairwise_Covariance)
        predicted = []
        for i in range(testSamples):
            if i%5000 == 0:
                print("Prediction for sample: ", i, "is in process.")
            Neighbors_pairwise_Covariance_inv = np.linalg.inv(Neighbors_pairwise_Covariance[i])
            Neighbors_Weights = Neighbors_pairwise_Covariance_inv.dot(test_neighors_Covariance[i].T)
            Neighbors_Weights = Neighbors_Weights.reshape(1,-1)
            Neighbors_Weights_exceptLangrange = Neighbors_Weights[0,:-1]
            WeightedSum_neighbors = np.sum(Neighbors_Weights_exceptLangrange * test_neighbors[i][:,2])
            predicted.append(WeightedSum_neighbors)
        predictedValues = np.asarray(predicted)
        return predictedValues
