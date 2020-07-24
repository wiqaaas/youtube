import numpy as np
from Fuzzy_Clustering import CLUSTERING

class REGRESSION(CLUSTERING):
    
    def __init__(self,sigma,n_clusters=2, max_iter=150, fuzzines=2, error=1e-5, 
                 random_state=42, dist="euclidean", method="Cmeans",outputCov=True):
        
        CLUSTERING.__init__(self,n_clusters=n_clusters, max_iter=max_iter, fuzzines=fuzzines, 
                            error=error, random_state=random_state, dist=dist, method=method)
        
        self.n_clusters = n_clusters
        self.fuzzines = fuzzines
        self.sigma = sigma
        self.outPutCov = outputCov 
        self.memb, self.class_centers = None, None
        
    def initiate_var(self,trainPoints):
        memberships, new_class_centers = self.fit(trainPoints)
        return memberships, new_class_centers

    def calculateFuzzyCov(self,data):
        memberships = self.memb
        new_class_centers = self.class_centers
        fuzzines = self.fuzzines
        n_clusters = self.n_clusters

        #calculating covariance matrix in its fuzzy form  
        fuzzyMem = memberships ** fuzzines
        n_clusters = n_clusters
        FcovInv_Class = []
        dim = data.shape[1]
        for i in range(n_clusters): 
            diff = data-new_class_centers[i]
            left = np.dot((fuzzyMem[:,i].reshape(-1,1)*diff).T,diff)/np.sum(fuzzyMem[:,i],axis=0)
            Fcov = (np.linalg.det(left)**(-1/dim))*left
            FcovInv = np.linalg.inv(Fcov)
            FcovInv_Class.append(FcovInv)
        return FcovInv_Class
    
    def antecedentMembership(self,data,regionTrain,testpoint):
        n_Clusters = self.n_clusters
        sigma = self.sigma
        covLst = self.calculateFuzzyCov(data)
        dist = np.zeros([regionTrain.shape[0],len(covLst)])
        delta = testpoint-regionTrain
        for i in range(n_Clusters):
            if self.outPutCov==True:
                DeltaSquare = (delta**2)*np.abs(covLst[i][[0,1],2])
            else:
                DeltaSquare = (delta**2)
            dist[:,i] = np.sqrt(np.sum(DeltaSquare,axis=1))
        antecedentMem = np.exp(-dist**2/(2*(sigma**2)))
        return antecedentMem
    
    def linearReg(self,data,regionTrain,regionTrainOut,testpoint):
        n_Clusters = self.n_clusters
        antecedentMem = self.antecedentMembership(data,regionTrain,testpoint)
        linearParam = np.zeros([n_Clusters,regionTrain.shape[1]+1])
        knownNew = np.concatenate((regionTrain,np.ones([regionTrain.shape[0],1])),axis=1)
        for i in range(n_Clusters):
            actMemDiag = np.diag(antecedentMem[:,i])
            XtransposeW = knownNew.T@actMemDiag
            linearParam[i] = np.linalg.inv(XtransposeW@knownNew)@(XtransposeW@regionTrainOut)
        return linearParam, antecedentMem
    
    def Pred(self,data,regionTrain,regionTrainOut,testpoint):
        n_Clusters = self.n_clusters
        linearParam, antecedentMem = self.linearReg(data,regionTrain,regionTrainOut,testpoint)
        pred_val = np.zeros(n_Clusters)
        for i in range(n_Clusters):
            pred_val[i] = np.sum(testpoint*linearParam[i,[0,1]])+linearParam[i,2]
        return pred_val, linearParam, antecedentMem

    def consequentMembership(self,data,regionTrain,regionTrainOut,testpoint,regionTrainMem):
        pred_val, linearParam, antecedentMem = self.Pred(data,regionTrain,regionTrainOut,testpoint)
        alpha = antecedentMem*regionTrainMem
        summation = np.sum(alpha,axis=0)
        total_influence_neighbors = np.sum(alpha)
        consequentMem = summation/total_influence_neighbors  
        return consequentMem, total_influence_neighbors, pred_val, linearParam, antecedentMem
    

    def fit_init(self,data,regionTrain,regionTrainOut,testpoint,regionTrainMem):
        
        consequentMem, total_influence_neighbors, pred_val, linearParam, antecedentMem = self.consequentMembership(data,regionTrain,regionTrainOut,testpoint,regionTrainMem)
        return consequentMem, total_influence_neighbors, pred_val, linearParam, antecedentMem
 
    def fit_final(self,dataTrain,regionTrain,regionTrainOut,regionTrainMem,regionTest):
        n_Clusters = self.n_clusters
        finalPredVal = np.zeros([regionTest.shape[0],n_Clusters])
        finalConsequentMem = np.zeros([regionTest.shape[0],n_Clusters])
        finalNeighborsInfluence = np.zeros([regionTest.shape[0],1])
        for sample in range(regionTest.shape[0]):
            testpoint = regionTest[sample]
            consequentMem, total_influence_neighbors, pred_val, linearParam, antecedentMem = self.fit_init(dataTrain,regionTrain,
                                            regionTrainOut,testpoint,regionTrainMem)
            finalPredVal[sample] = pred_val
            finalConsequentMem[sample] = consequentMem
            finalNeighborsInfluence[sample] = total_influence_neighbors
            print("Prediction for sample: ",sample," has been done.")
        return finalPredVal, finalConsequentMem, finalNeighborsInfluence
    
    def fit_regression(self,X_train=None,y_train=None,X_test=None,Local=False,regionTrain=None,regionTrainOut=None,regionTest=None,regionTrainIndex=None):

        if (len(y_train.shape))==1:
            dataTrain = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        else:
            dataTrain = np.concatenate((X_train,y_train),axis=1)
        
        self.memb, self.class_centers = self.initiate_var(dataTrain)
        
        if Local==False:
            regionTrain=X_train
            regionTrainOut=y_train
            regionTest=X_test
            regionTrainMem=self.memb
            ########
            finalPredVal, finalConsequentMem, finalNeighborsInfluence = self.fit_final(dataTrain,regionTrain,regionTrainOut,
            regionTrainMem,regionTest)
            finalPred = np.sum((finalPredVal*finalConsequentMem),axis=1)/np.sum(finalConsequentMem,axis=1)
        else:
            regionTrainMem = self.memb[regionTrainIndex]
            ########
            finalPredVal, finalConsequentMem, finalNeighborsInfluence = self.fit_final(dataTrain,regionTrain,regionTrainOut,
            regionTrainMem,regionTest)
            finalPred = np.sum((finalPredVal*finalConsequentMem),axis=1)/np.sum(finalConsequentMem,axis=1)
        return finalPred