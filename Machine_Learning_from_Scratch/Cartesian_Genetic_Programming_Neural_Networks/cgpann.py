"""
Created on Thu Nov 14 18:48:46 2019
@author: waqas
"""
import random
import numpy as np
from math import ceil, exp
from copy import deepcopy
from sklearn.metrics import mean_squared_error


class cgpann(object):
    
    """
    Cartesian Genetic Programming of Artificial Neural Networks 
    is a NeuroEvolutionary method. 
    
    The implementation of cgpann in this class is done such that 
    1) whole network is stored in one array/list/genotype.
    2) only mutation is done for evolution. however, code for crossover is written but 
    commented out for anyone to try themselves in case they need.
    
    In next versions, crossover will be done.
    recurrent version will also be released in future.
    plasticity will also be introduced in next versions.
    
    ...
    Attributes
    ..........
    num_features : int
        total number of input features/attributes
    num_outputs : int
        total number of output features/attributes
    nodes : int
        total number of nodes/neurons in hidden layer (default 100)
    connections : int
        total number of allowed connections between neurons (default 2)
    generations : int
        number of iterations that algorithm needs to run (default 1000)
    offsprings : int 
        genetic algorithm hyperparameter
        number of childs per generation/population (default 10)
    mutationRate : float between 0 to 1
        genetic algorithm hyperparameter
        rate of change/mutation/randomness in each genotype/child (default 0.1)
    learning_rate : float between 0 to 1
        learning rate is multiplied with mutated values during mutation (default 0.1)
    outMutationChance : float between 0 to 1
        outMutationChance means probability of output nodes directly getting mutated
        This is used to increase the speed of convergence (default 0.2)
    nodesPerOut : int
        nodesPerOut is number of nodes that will be used for each output attribute 
        final value of each attribute will be average of those nodes values (default 2)
    functions : int 1 or 2 or 3
        possibilities of activation functions to be used
        1 means only sigmoid, 2 means both sigmoid and tanh, 3 means all 
        sigmoid, tanh, and relu (default 1)
        if gene_type = 1 i.e. simple cgp then values could be 1 or 2
        1 means or function and 2 means both or and 'and' function
    gene_type : int 1 or 2 or 3
        type of cgpann
        1 means basic cgp i.e. each node will be having only connection and function
        2 means basic cgpann i.e. each node will be having connection, weight, and function
        3 means full cgpann i.e. each node will be having connection, weight, switch, and function (default 2)
    seed : int 
        random seed number for reproducibility (default 42)
    verbose : bool True or False
        if verbose is True then prints correlation for each iteration on train data (default True)
    
    ...
    Methods
    .......
    Genotype()
    It outputs a network having all values randomly intiailized.
    
    initPopulation()
    It outputs list containing multiple networks i.e. equal to number of offsprings 
    
    ActiveGenesNoInp()
    Given genotype it outputs active nodes in it which is further used for prediction 
    It outputs two lists:
    First list contains both input and active nodes in genotype/network (given as input).
    second list contains only active nodes in genotype/network (given as input).
    
    mutation()
    Given genotype it outputs mutated genotype
    
    Func()
    a function which calls relevant appropriate activation function
    
    logicFunc()
    a function which calls relevant digital logic function in case of only cgp i.e. gene_type=1
    
    evaluateGeneral()
    It outputs predicted value given input data array and best genotype
    It does that efficiently by first finding active nodes
    
    evaluateGeneralNoActive()
    It outputs predicted value given input data array and best genotype
    It is very time consuming hence use evaluateGeneral instead
    
    fit()
    It outputs bestgeno and it also saves best genotype found during training locally.
    further if verbose is true it pritns results for each iteration
    
    predict()
    It outputs predicted value for given input
    
    """
    
    def __init__(self, num_features, num_outputs, nodes=20, connections=3, generations=100,
                 offsprings=10, mutationRate=0.1, learning_rate=0.1,outMutationChance=0.2,
                 nodesPerOut=2, functions=1, gene_type=2, seed=42, verbose=True):
        
        self._inp = num_features
        self._systemOut = num_outputs
        self._out = nodesPerOut*num_outputs
        self._seed = seed
        self._nodes = nodes
        self._connections = connections
        self._functions = functions
        self._gene_type = gene_type
        self._offsprings = offsprings
        self._mutationRate = mutationRate
        self._generations = generations
        self._learning_rate = learning_rate
        self._outMutationChance = outMutationChance
        self._verbose = verbose 
        self._range_node = connections*gene_type+1
        
    def GenoType(self): 
        '''
        Parameters
        ----------
        None
        
        '''
        
        inp = self._inp 
        out = self._out 
        nodes = self._nodes 
        connections = self._connections 
        functions = self._functions 
        gene_type = self._gene_type 
        
        nodeFunctions=[random.randint(0,functions-1) for x in range(nodes)]
        GenoType = [None for _ in range(nodes*(connections*gene_type+1)+out)] 
        jj = 0
        for i in range(nodes):
            for j in range(0,connections*gene_type,gene_type):
                if gene_type == 1:
                    GenoType[i*(connections*gene_type+1)+j] = random.randint(0,inp+i-1) #CONNECTION
                elif gene_type == 2:
                    GenoType[i*(connections*gene_type+1)+j] = random.randint(0,inp+i-1) #CONNECTION
                    GenoType[i*(connections*gene_type+1)+j+1] = random.uniform(-1,1) #WEIGHT OF CONNECTION
                elif gene_type == 3:
                    GenoType[i*(connections*gene_type+1)+j] = random.randint(0,inp+i-1) #CONNECTION
                    GenoType[i*(connections*gene_type+1)+j+1] = random.uniform(-1,1) #WEIGHT OF CONNECTION
                    GenoType[i*(connections*gene_type+1)+j+2] = random.randint(0,1) #SWITCH OF CONNECTION
                jj = j+gene_type
            GenoType[i*(connections*gene_type+1)+jj] = nodeFunctions[i]
        for i in range(out):
            GenoType[nodes*(connections*gene_type+1)+i] = random.randint(0,inp+nodes-1)
        return GenoType
    
    def initPopulation(self):
        '''
        Parameters
        ----------
        None
        
        '''

        offsprings = self._offsprings
        population = []
        for i in range(offsprings+1):
            population.append(self.GenoType())
        return population
    
    def ActiveGenesNoInp(self,bestGeno):
        '''
        Parameters
        ----------
        bestGeno : list
        It accepts list containing network/genotype 
        
        '''

        range_node = self._range_node
        out = self._out
        nodes = self._nodes
        inp = self._inp
        gene_type = self._gene_type
        genotype = bestGeno
        
        activeNodes = []
        for i in range(out):
            activeNodes.append(genotype[nodes*(range_node)+i])    
        activeNodesWithoutInp = [item-inp for item in activeNodes]    
        activeNodesWithoutInp.sort(reverse=True)
    
        count = 0
        val = activeNodesWithoutInp[count]
        while(val>=0):
            nodeNum = val
            nodeStart = nodeNum*range_node
            for ii in range(0,range_node,gene_type):
                activeNodesWithoutInp.append(genotype[nodeStart+ii]-inp)
            activeNodesWithoutInp = list(set(activeNodesWithoutInp))
            activeNodesWithoutInp.sort(reverse=True)
            count = count+1
            val = activeNodesWithoutInp[count]         
            
        activeNodesWithInp = [item+inp for item in activeNodesWithoutInp] #activeNodes with input at 0 index
        activeNodesWithInp = list(set(activeNodesWithInp))
        activeNodesWithInp.sort(reverse=False)
        
        activeNodesWithoutInp = [item for item in activeNodesWithoutInp if item >=0] #actuveNodes without input 
        activeNodesWithoutInp.sort(reverse=False)
        
        return activeNodesWithInp, activeNodesWithoutInp  
    
    def mutation(self, parentGeno):
        '''
        Parameters
        ----------
        parentGeno : list
        It outputs genotype after mutating it
        
        '''

        inp = self._inp
        out  = self._out 
        nodes = self._nodes
        range_node = self._range_node
        functions = self._functions
        gene_type = self._gene_type 
        mutationRate = self._mutationRate 
        learning_rate = self._learning_rate
        outMutationChance = self._outMutationChance
        
        select = random.uniform(0,1)
        if select <= outMutationChance:
            genesMutatedOut = ceil(mutationRate*out)
            rndNumOut = [None for i in range(genesMutatedOut)]
            rndNumOut = [random.randint(1,out) for i in range(genesMutatedOut)]
            for i in rndNumOut:
                parentGeno[nodes*range_node+i-1] = random.randint(0,inp+nodes-1)  
        else:
            sizeGeno = len(parentGeno)-out
            genesMutated = ceil(mutationRate*sizeGeno)
            rndNum = [random.randint(1,sizeGeno) for i in range(genesMutated)]
            for i in range(len(rndNum)):               
                nodeNo, remainder = divmod(rndNum[i],range_node)
                remainder1 = remainder%3
                if gene_type==1:
                    if remainder==0: #function
                        parentGeno[rndNum[i]-1] = random.randint(0,functions-1)
                    else:   #connection
                        parentGeno[rndNum[i]-1] = random.randint(0,inp+nodeNo-1)  
                elif gene_type==2:
                    if remainder==0: #function
                        parentGeno[rndNum[i]-1] = random.randint(0,functions-1)
                    elif remainder % 2 == 0:    #weights
                        parentGeno[rndNum[i]-1] += learning_rate*random.uniform(-1,1)
                    else:   #connection
                        parentGeno[rndNum[i]-1] = random.randint(0,inp+nodeNo-1)                
                elif gene_type==3:
                    if remainder==0:    #function
                        parentGeno[rndNum[i]-1] = random.randint(0,functions-1)
                    elif remainder1==0:   #switch
                        parentGeno[rndNum[i]-1] = random.randint(0,1)
                    elif remainder1==1: #connection
                        parentGeno[rndNum[i]-1] = random.randint(0,inp+nodeNo-1)
                    elif remainder1==2: #weights
                        parentGeno[rndNum[i]-1] = random.uniform(-1,1)
        return parentGeno
 
    @staticmethod
    def Func(summation,typeFunc):
        '''
        Parameters
        ----------
        summation : float
        typeFunc : int 0 or 1 or 2
        0 means sigmoid, 1 means tanh, 2 means relu
        
        '''

        if typeFunc == 0:
            funcVal = (1/(1+exp(-summation)))
        elif typeFunc == 1:
            funcVal = (2/(1+exp(-2*summation)))-1
        elif typeFunc == 2:
            funcVal = max(summation,0)
        return funcVal

    @staticmethod
    def logicFunc(connections,funcVal,array1):
        '''
        Parameters
        ----------
        summation : float
        funcVal : int 0 or 1 
        0 means or, 1 means and
        
        '''
       
        if funcVal == 0: 
            funcOR = 0
            for i in range(connections):
                funcOR = array1[i] or funcOR
        if funcVal == 1: 
            funcAND = 1
            for i in range(connections):
                funcAND = array1[i] and funcAND
    
    def evaluateGeneral(self,x,genotype):
        '''
        Parameters
        ----------
        x : array 
        array of input values
        genotype : list
        list containing network
        '''
        
        inp = self._inp 
        out = self._out
        nodes = self._nodes
        connections = self._connections
        range_nodes = self._range_node
        gene_type = self._gene_type
        systemOut = self._systemOut
        
        _ , activeNodesNoInp = self.ActiveGenesNoInp(genotype) 
        
        arr = np.zeros([x.shape[0],inp+nodes])
        for i in range(x.shape[1]):
            arr[:,i] = x[:,i]
        for j in activeNodesNoInp:
            funcVal = genotype[j*(range_nodes)+(range_nodes-1)]
            for i in range(0,x.shape[0]):
                if gene_type == 1: 
                    array1 = np.zeros(connections)
                    for ii in range(0,connections,1):
                        array1[ii] = arr[i,genotype[j*range_nodes+ii]]
                    arr[i,inp+j] = self.logicFunc(connections,funcVal,array1)
                else:    
                    summation = 0
                    for ii in range(0,connections,gene_type):
                        if gene_type==2:
                            summation += arr[i,genotype[j*range_nodes+ii]]*genotype[j*range_nodes+ii+1]
                        elif gene_type == 3:
                            summation += arr[i,genotype[j*range_nodes+ii]]*genotype[j*range_nodes+ii+1]*genotype[j*range_nodes+ii+2]       
                    arr[i,inp+j] = self.Func(summation,funcVal)
                    
        y_pred = np.zeros([x.shape[0],out])
        for i in range(out):
            y_pred[:,i] = arr[:,genotype[-i-1]]
        
        y_pred_ave = np.zeros([x.shape[0],systemOut])
        outputNodes = out/systemOut
        for i in range(systemOut):
            y_pred_ave[:,i] = np.mean(y_pred[:,int(i*outputNodes):int((i+1)*outputNodes)],axis=1)
            
        return y_pred_ave
    
    def evaluateGeneralNoActive(self,x,genotype):
        '''
        Parameters
        ----------
        x : array 
        array of input values
        genotype : list
        list containing network
        '''
        
        inp = self._inp 
        out = self._out
        nodes = self._nodes
        connections = self._connections
        range_nodes = self._range_node
        gene_type = self._gene_type
        systemOut = self._systemOut
            
        arr = np.zeros([x.shape[0],inp+nodes])
        for i in range(x.shape[1]):
            arr[:,i] = x[:,i]
        for j in range(nodes):
            funcVal = genotype[j*(range_nodes)+(range_nodes-1)]
            for i in range(0,x.shape[0]):
                if gene_type == 1: 
                    array1 = np.zeros(connections)
                    for ii in range(0,connections,1):
                        array1[ii] = arr[i,genotype[j*range_nodes+ii]]
                    arr[i,inp+j] = cgpann.logicFunc(connections,funcVal,array1)
                else:    
                    summation = 0
                    for ii in range(0,connections,gene_type):
                        if gene_type==2:
                            summation += arr[i,genotype[j*range_nodes+ii]]*genotype[j*range_nodes+ii+1]
                        elif gene_type == 3:
                            summation += arr[i,genotype[j*range_nodes+ii]]*genotype[j*range_nodes+ii+1]*genotype[j*range_nodes+ii+2]       
                    arr[i,inp+j] = cgpann.Func(summation,funcVal)
                    
        y_pred = np.zeros([x.shape[0],out])
        for i in range(out):
            y_pred[:,i] = arr[:,genotype[-i-1]]
        
        y_pred_ave = np.zeros([x.shape[0],systemOut])
        outputNodes = out/systemOut
        for i in range(systemOut):
            y_pred_ave[:,i] = np.mean(y_pred[:,int(i*outputNodes):int((i+1)*outputNodes)],axis=1)
            
        return y_pred_ave

    def fit_data(self,X_train,y_train):
        '''
        Parameters
        ----------
        X_train : array
        array of input train data 
        y_train : array
        array of output train data
        '''

        generations = self._generations
        offsprings = self._offsprings
        Population = self.initPopulation()
        bestGeno = Population[0]
        for iteration in range(generations):    
            parentGeno = deepcopy(bestGeno)
            for i in range(1,offsprings):
                Population[i] = self.mutation(parentGeno)
            ErrorCorrelation = []     
            for i in range(offsprings+1):
                y_pred = self.evaluateGeneral(X_train,Population[i])
                err = mean_squared_error(y_train,y_pred)
                ErrorCorrelation.append(err)    
            bestGenIndex = np.argmin(ErrorCorrelation)
            if self._verbose:    
                print("Generation number: ", iteration+1, " is having mse: ",ErrorCorrelation[bestGenIndex])
            bestGeno = deepcopy(Population[bestGenIndex])
            Population[0] = deepcopy(bestGeno)
        
        self.GenoPrediction = bestGeno
        return bestGeno

    def predict_data(self,x):
        '''
        Parameters
        ..........
        x : array
        array of test data
        '''
        return self.evaluateGeneral(x,self.GenoPrediction)
    
    
    
# Crossover Implementation
    
# =============================================================================
#     def CrossOver(self,genotype):  
#         
#         crossoverPercent = self.crossoverPercent
#         nodes = self.nodes
#         range_nodes = self.range_node
#         
#         crossover = []
#         for indi,i in enumerate(genotype):
#             crossover.append(deepcopy(i))
#             for indii,ii in enumerate(genotype):  
#                 if indi >= indii+1:
#                     continue
#                 lst1 = deepcopy(i)
#                 lst11 = deepcopy(ii)
#                 lst2 = deepcopy(i)
#                 lst22 = deepcopy(ii)
#                 lst1[:int(crossoverPercent*nodes)*range_nodes] = lst11[:int(crossoverPercent*nodes)*range_nodes]
#                 crossover.append(lst1)
#                 lst2[int(crossoverPercent*nodes)*range_nodes:] = lst22[int(crossoverPercent*nodes)*range_nodes:]
#                 crossover.append(lst2)
#         return crossover
#     
#     def initializeNetworks(self):
#         networks = []
#         for i in range(self.network):
#             #initiate population
#             Population = self.initPopulation()
#             networks.append(Population)
#         return networks
# 
#     def initializeGenotype(self):
#         network = self.network
#         networks = self.initializeNetworks()
#         
#         bestGeno = []
#         for i in range(network):
#             bestGeno.append(networks[i][0])
# 
#     def fit_crossover(self,X_train,y_train):
#         networks = self.initializeNetworks()
#         bestGeno = self.initializeGenotype()
#         offsprings = self.offsprings
#         
#         for iteration in range(100*2):                                   
#             if iteration < 100:
#                     
#                 for i,Population in enumerate(networks):    
#                     parentGeno = deepcopy(bestGeno[i])
#                     Population[0] = deepcopy(bestGeno[i])
#                     for ii in range(1,offsprings+1):
#                         mut = self.mutation(parentGeno)        
#                         Population[ii] = deepcopy(mut)
#          
#             #######################
#             
#                     PopulationCorrelation = []     
#                     for j in range(len(Population)):
#                         #activeNodesInp,activeNodesNoInp = self.ActiveGenesNoInp(Population[j]) 
#                         y_pred = self.evaluateGeneral(X_train,Population[j])
#                         correlation = pearsonr(y_pred[:,0],y_train[:,0])
#                         PopulationCorrelation.append(correlation[0])
#                         bestGenIndex = np.argmax(PopulationCorrelation)
#                         bestGeno[i] = deepcopy(Population[bestGenIndex])
#                                        
#                     print(i,"correlation after simple mutation is: ",PopulationCorrelation[bestGenIndex]) 
#                 print(iteration,"_th generation")
#             else:    
#                 crossover = self.CrossOver(bestGeno)
#                 
#                 mutationDuration = 100
#                 for iMut in range(mutationDuration):
#                     for i,lst in enumerate(crossover):
#                         crossoverPop = []
#                         parentGeno = deepcopy(lst)
#                         crossoverPop.append(deepcopy(parentGeno))
#                         for ii in range(1,offsprings+1):
#                             mut = self.mutation(parentGeno) 
#                             crossoverPop.append(deepcopy(mut))
#                         
#                         PopulationCorrelation = [] 
#                         for j in range(len(crossoverPop)):
#                             #activeNodesInp,activeNodesNoInp = self.ActiveGenesNoInp(crossoverPop[j]) 
#                             y_pred = self.evaluateGeneral(X_train,crossoverPop[j])
#                             correlation = pearsonr(y_pred[:,0],y_train[:,0])
#                             PopulationCorrelation.append(correlation[0])
#                             bestGenIndex = np.argmax(PopulationCorrelation)
#                             bestGeno[i] = deepcopy(Population[bestGenIndex])
#                             crossPopBest =  deepcopy(crossoverPop[bestGenIndex])
#                             #print(j, "after mutation",correlation[0])
#                                     
#                         crossover[i] = deepcopy(crossPopBest)
#                         print(i,"number crossover best correlation after mutation is: ",PopulationCorrelation[bestGenIndex])
#                     print(iMut,"_th mutation cycle") 
#                 
#                 bestCrossover = []
#                 for j in range(len(crossover)):  
#                     #activeNodesInp,activeNodesNoInp = self.ActiveGenesNoInp(crossover[j]) 
#                     y_pred = self.evaluateGeneral(X_train,crossover[j])
#                     correlation = pearsonr(y_pred[:,0],y_train[:,0])
#                     bestCrossover.append(correlation[0])
#                 bestIndex = np.argmax(bestCrossover)       
#                 print("########################################################")
#                 print(iteration, "_th generation best correlation: ",bestCrossover[bestIndex])
#                
#                 bestInd = sorted(range(len(bestCrossover)), key=lambda i: bestCrossover[i], reverse=True)[:self.network]
#                 for i in range(len(bestInd)):
#                     lst = crossover[bestInd[i]]
#                     bestGeno[i] = deepcopy(lst)
# =============================================================================
