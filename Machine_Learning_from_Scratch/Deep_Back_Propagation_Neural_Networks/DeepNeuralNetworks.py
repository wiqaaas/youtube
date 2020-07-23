
"""
This Python file contains code for implementation of Back Propagation
Neural Network using Numpy.

"""

import numpy as np
import matplotlib.pyplot as plt

class DNN:
    """
    Back Propagation Neural Network Implementation using Numpy
    
    Attributes
    ----------
    layer_dims : list
        list containing the number of neurons/nodes for 
        hidden layers and ouput layer e.g. 
        [256,128,64,1] means 3-layer neural network having 1 output
    num_iterations : int
        total number of iterations to be run
    learning_rate : float
        the learning rate alpha to be used 
    weight_initialization : str
        type of weight initialization strategy to be used e.g. 
        "He" or "random"
    internal_activation : str
        type of activation function to be used for hidden layers e.g.
        "relu" or "sigmoid"
    output_activation  : str
        type of activation function to be used for output layer
    L2_regularization : Boolean
        if True then l2 regularization is applied to cost function
    lambd : float
        amount of l2 regularization to be applied
    dropout : Boolean
        if True then dropout regularization is applied
        (But this currently is unstable, as it sometimes generate nan values)
    keep_prob : float
        amount of dropout regularization 
        (probability of keeping the nodes alive)
    optimization : str
        type of optimization function to be used (default GD) e.g.
        "GD" gradient descent or
        "GD_Momentum" gradient descent with momentum 
        (Currently it does not work)
        "RMSprop" 
        (Currently it does not work)
    alpha : float
        alpha used in case of optimization is other than GD (default 0.9)
    eps : float
        eps used in case of optimization is other than GD (default 0.0001)
    seed : int
        random seed number for reproducibility of results (default 42)
    verbose : Boolean
        if verbose is True then it outputs the intermediate results on screen.
    optimum_parameters : dict
        in case parameters are known then (dict), if not then (None)
        dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
        Here 1,2,3 are hidden layer number.
        Shape of numpy array for W1 will be (hidden layer1, input layer)
        Shape of numpy array for b1 will be (hidden layer1, 1)
        Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
        Shape of numpy array for b2 will be (hidden layer2, 1)
        Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
        Shape of numpy array for b3 will be (hidden layer3, 1)
        and so on...
        
    Methods
    -------
    fit(self,X,y)               
        Given input and output along with different options set and defined during
        making of class object,
        This function returns optimized weight and bias parameters for neural network.

        Parameters
        ----------
        X : array
            input array of shape (input features, samples)
        Y : array
            output array of shape (output features, samples)

        Returns
        -------
        parameters : dict
            updated parameters
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        
    predict(self,X, parameters)
        Given training data input and optimized parameters,
        This function returns predicted value
        
        Parameters
        ----------
        X : array
            input array of shape (input features, samples)
        parameters : dict
            updated parameters
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...

        Returns
        -------
        p : array
            final prediction values using optimized parameters

    """   
    
    def __init__(self,layer_dims,num_iterations,learning_rate,weight_initialization,
                   internal_activation, output_activation,
                   L2_regularization,lambd,dropout,keep_prob,
                   optimization="GD", alpha=0.9, eps=0.0001,
                   seed=42,verbose=True,optimum_parameters=None):
        """
        Initializes the class with input options and parameters for object creation
        
        Parameters
        ----------
        layer_dims : list
            list containing the number of neurons/nodes for 
            hidden layers and ouput layer e.g. 
            [256,128,64,1] means 3-layer neural network having 1 output
        num_iterations : int
            total number of iterations to be run
        learning_rate : float
            the learning rate alpha to be used 
        weight_initialization : str
            type of weight initialization strategy to be used e.g. 
            "He" or "random"
        internal_activation : str
            type of activation function to be used for hidden layers e.g.
            "relu" or "sigmoid"
        output_activation  : str
            type of activation function to be used for output layer
        L2_regularization : Boolean
            if True then l2 regularization is applied to cost function
        lambd : float
            amount of l2 regularization to be applied
        dropout : Boolean
            if True then dropout regularization is applied
            (But this currently is unstable, as it sometimes generate nan values)
        keep_prob : float
            amount of dropout regularization 
            (probability of keeping the nodes alive)
        optimization : str
            type of optimization function to be used (default GD) e.g.
            "GD" gradient descent or
            "GD_Momentum" gradient descent with momentum 
            (Currently it does not work)
            "RMSprop" 
            (Currently it does not work)
        alpha : float
            alpha used in case of optimization is other than GD (default 0.9)
        eps : float
            eps used in case of optimization is other than GD (default 0.0001)
        seed : int
            random seed number for reproducibility of results (default 42)
        verbose : Boolean
            if verbose is True then it outputs the intermediate results on screen.
        optimum_parameters : dict
            in case parameters are known then (dict), if not then (None)
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...

        Returns
        -------
        None
        """
        self.layer_dims = layer_dims
        self.weight_initialization = weight_initialization
        self.seed = seed
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.internal_activation = internal_activation
        self.output_activation = output_activation
        self.L2_regularization = L2_regularization
        self.lambd = lambd
        self.dropout = dropout # Dropout if true could sometimes result into nan values
        self.keep_prob = keep_prob
        self.optimum_parameters=optimum_parameters
        self.optimization = optimization
        self.alpha = alpha
        self.eps = eps
            
    @staticmethod
    def _initialize_parameters_deep(layers_dims,weight_initialization,seed):
        """
        Randomly initializes parameters
        
        Parameters
        ----------
        layers_dims : list
            list containing the number of neurons/nodes for 
            hidden layers and ouput layer e.g. 
            [256,128,64,1] means 3-layer neural network having 1 output
        weight_initialization : str
            type of weight initialization strategy to be used e.g. 
            "He" or "random"
        seed : int
            random seed number for reproducibility of results (default 42)

        Returns
        -------
        parameters : dict
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        """
        np.random.seed(seed)
        parameters = {}
        # L is number of layers in the network
        L = len(layers_dims)            
        for l in range(1, L):
              if weight_initialization == "He":
                    parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*np.sqrt(2 /layers_dims[l - 1])
                    parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
              elif weight_initialization == "random":
                    parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])*0.01
                    parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
  
        return parameters
      
    @staticmethod
    def _linear_forward(pre_Activation, current_W, current_b):     
        """
        Given previous layer activation, and current layer parameters, 
        returns current layer linear transformation
        
        Parameters
        ----------
        pre_Activation : array
            Output of previous layer after applying activation function i.e. g(z) or g(wx+b)
        current_W : array
            Weight of current layer
        current_b : array
            Bias of current layer

        Returns
        -------
        current_Z : array
            Output of current layer before activation applied i.e. z or wx+b
        cache_preActivation_currentW_currentB : tuple
            caches a tuple containing 
            Output of previous layer after applying activation function i.e. g(z) or g(wx+b)
            Weight and bias of current layer
        """
        current_Z = np.dot(current_W,pre_Activation) + current_b #A in first layer will be X
        assert(current_Z.shape == (current_W.shape[0], pre_Activation.shape[1]))
        cache_preActivation_currentW_currentB = (pre_Activation, current_W, current_b) #This cache consists of all inputs to layer
        return current_Z, cache_preActivation_currentW_currentB

    @staticmethod
    def _sigmoid(current_Z):
        """
        Given layer linear transformation,
        it returns result after applying sigmoid activation function
        
        Parameters
        ----------
        current_Z : array
            current layer linear transformed values i.e. z or wx+b or
            output of current layer before activation function applied

        Returns
        -------
        current_Activation : array
            output of current layer after applying sigmoid activation function
        cache_currentZ : array
            caches output of current layer before applying activation function

        """
        current_Activation = 1/(1+np.exp(-current_Z))
        cache_currentZ = current_Z    
        return current_Activation, cache_currentZ
      
    @staticmethod
    def _relu(current_Z):
        """
        Given layer linear transformation,
        it returns result after applying relu activation function
        Parameters
        ----------
        current_Z : array
            current layer linear transformed values i.e. z or wx+b or
            output of current layer before activation function applied

        Returns
        -------
        current_Activation : array
            output of current layer after applying relu activation function
        cache_currentZ : TYPE
            caches output of current layer before applying activation function
        """
        current_Activation = np.maximum(0,current_Z)
        assert(current_Activation.shape == current_Z.shape)
        cache_currentZ = current_Z 
        return current_Activation, cache_currentZ

    def _linear_activation_forward(self,pre_Activation, current_W, current_b, activation):
        """
        Given previous layer output, current layer weights and bias, and type of 
        activation function 
        it returns current layer output after applying activation function
        and caches all other values
        
        Parameters
        ----------
        pre_Activation : array
            previous layer output after applying activation function.
        current_W : array
            current layer weights.
        current_b : array
            current layer bias.
        activation : str
            type of activation function to be applied for current hidden layer

        Returns
        -------
        current_Activation : array
            output of current layer after applying activation function
        cache_preActivation_currentW_currentB_currentZ : tuple
            caches previous layer output after applying activation function,
            current layer weights and bias,
            current layer linear transformation before applying activation function
        """
        current_Z, cache_preActivation_currentW_currentB = self._linear_forward(pre_Activation, current_W, current_b)
          
        if activation == "sigmoid":
              current_Activation, cache_currentZ = self._sigmoid(current_Z)
          
        elif activation == "relu":
              current_Activation, cache_currentZ = self._relu(current_Z)
          
        assert (current_Activation.shape == (current_W.shape[0], pre_Activation.shape[1]))
        cache_preActivation_currentW_currentB_currentZ = (cache_preActivation_currentW_currentB, cache_currentZ) 
        #linear_cache stores input and activation_cache stores output of this layer
        
        return current_Activation, cache_preActivation_currentW_currentB_currentZ

    def _L_model_forward(self,X, parameters,dropout,keep_prob,internal_activation,output_activation):
        """
        A function applying complete feed forward step from input to output.
        In the process it also caches all information which will be later used for 
        back propagation.
        
        Parameters
        ----------
        X : array
            input array of shape (features, samples)
        parameters : dict
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        dropout : Boolean
            if True then dropout regularization is applied
        keep_prob : float
            amount of dropout regularization 
            (probability of keeping the nodes alive)
        internal_activation : str
            type of activation function for hidden layer
        output_activation : str
            type of activation function for output layer

        Returns
        -------
        output : array
            final predicted value for input array after feed forward 
        LST_cache_preActivation_currentW_currentB_currentZ : list
            list of tuples containing cache values. 
            For each layer each layer tuple inside list contains previous layer 
            output after applying activation function, current layer weights, bias, 
            and current layer linear transformation before activation function
        cache_dropout : list
            list cotaining array for each layer.
            For each layer the information of which nodes were kept during random
            dropout regularization was cached in form of binary arrays having 1 or 0 values
        """
        cache_dropout = [] #to implement dropout
        LST_cache_preActivation_currentW_currentB_currentZ = []
        L = len(parameters) // 2    # number of layers in the neural network 
        # it is divided by two because parameters consist of both w and b for each layeR
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        current_Activation = X
        #first element of list L is basically input features
        for l in range(1, L):
            pre_Activation = current_Activation
            current_W = parameters['W' + str(l)] 
            current_b = parameters['b' + str(l)]
            activation = internal_activation
            current_Activation, cache_preActivation_currentW_currentB_currentZ = self._linear_activation_forward(pre_Activation, current_W, current_b, activation)
           
            #dropout regularization
            if dropout:
                tmp = np.random.rand(current_Activation.shape[0],current_Activation.shape[1])
                tmp = tmp < keep_prob  # Step 2: convert entries of tmp to 0 or 1 (using keep_prob as the threshold)
                cache_dropout.append(tmp)
                current_Activation = np.multiply(current_Activation, tmp) # Step 3: shut down some neurons of current_Activation
                current_Activation /= keep_prob # Step 4: scale the value of neurons that haven't been shut down
      
            #store cache
            LST_cache_preActivation_currentW_currentB_currentZ.append(cache_preActivation_currentW_currentB_currentZ)
              
        # This following step is for last L layer 
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        pre_Activation = current_Activation
        current_W = parameters['W' + str(L)] 
        current_b = parameters['b' + str(L)]
        activation = output_activation
        current_Activation, cache_preActivation_currentW_currentB_currentZ = self._linear_activation_forward(pre_Activation, current_W, current_b, activation)
        #store cache
        LST_cache_preActivation_currentW_currentB_currentZ.append(cache_preActivation_currentW_currentB_currentZ)
  
        output = current_Activation
      
        assert(output.shape == (1, X.shape[1]))
      
        return output, LST_cache_preActivation_currentW_currentB_currentZ, cache_dropout
      
    @staticmethod
    def _cross_entropy_cost(output, Y, parameters, lambd, L2_regularization):          
        """
        Given predicted output, actual output, parameters, l2 regularization values,
        it returns the cross entropy cost values for final layer.

        Parameters
        ----------
        output : array
            final predicted values after feed forward step
        Y : array
            actual output values
        parameters : dict
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        lambd : float
            amount of l2 regularization to be applied
        L2_regularization : Boolean
            if True then l2 regularization is applied to cost function
        
        Returns
        -------
        cross_entropy_cost : array
            Array containing cross entropy cost values for output layer
        """
        ##following line was added to remove nan values during dropout, as logloss func was used
        #output = np.clip(output, 1e-5, 1. - 1e-5)                
        m = Y.shape[1]
        # Compute loss from output and y
        cross_entropy_cost = (-1 / m) * np.sum(np.multiply(Y, np.log(output)) + np.multiply(1 - Y, np.log(1 - output)))
        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cross_entropy_cost = np.squeeze(cross_entropy_cost)      
        
        ##General Note about L1 vs L2:
        #L1 penalty or lasso regression is better than l2 penalty or ridge regression 
        #to remove certain features because it makes weight of certain features to zero.
        if L2_regularization:
            L = len(parameters)//2
            m = Y.shape[1]
            val = 0
            for l in range(1,L+1):
                  val += np.sum(np.square(parameters["W"+str(l)]))
            L2_regularization_cost = (1 / m)*(lambd / 2) * val
            cross_entropy_cost += L2_regularization_cost
        
        assert(cross_entropy_cost.shape == ())
        return cross_entropy_cost

    @staticmethod
    def _sigmoid_backward(derivative_postActivation, cache_currentZ): 
        """
        Finding derivative with repect to sigmoid activation function

        Parameters
        ----------
        derivative_postActivation : array
            array containing values representing derivative of loss function upto current layer 
        cache_currentZ : array
            current layer linear transformation i.e. z or wx+b 
            (cached during feed forward step)

        Returns
        -------
        derivative_currentZ : array
            array containing values representing derivative of loss upto
            current layer linear transformation 
        """
        tmp = 1/(1+np.exp(-cache_currentZ))
        derivative_postActivation_with_currentZ = tmp * (1-tmp)
        derivative_currentZ = derivative_postActivation * derivative_postActivation_with_currentZ    
        #where derivative_post_Activation is derivative of loss function upto this layer 
        assert (derivative_currentZ.shape == cache_currentZ.shape) 
        return derivative_currentZ

    @staticmethod
    def _relu_backward(derivative_postActivation, cache_currentZ):
        """
        Finding derivative with repect to relu activation function

        Parameters
        ----------
        derivative_postActivation : array
            array containing values representing derivative of loss function upto current layer 
        cache_currentZ : array
            current layer linear transformation i.e. z or wx+b 
            (cached during feed forward step)

        Returns
        -------
        derivative_currentZ : array
            array containing values representing derivative of loss upto
            current layer linear transformation 
        """
        derivative_currentZ = np.array(derivative_postActivation, copy=True) 
        #where derivative_post_Activation is derivative of loss function upto this layer
        derivative_currentZ[cache_currentZ <= 0] = 0
        assert (derivative_currentZ.shape == cache_currentZ.shape)
        return derivative_currentZ

    def _linear_activation_backward(self,derivative_postActivation,cache_preActivation_currentW_currentB_currentZ,
                                   activation,samples,lambd,L2_regularization): 
        """
        Given derivative values upto next layer,
        It returns derivative values upto previous layer,
        It also returns derivative with repect to current layer weights and bias

        Parameters
        ----------
        derivative_postActivation : array
            array containing values representing derivative of loss function upto current layer
        cache_preActivation_currentW_currentB_currentZ : tuple
            tuple containing Output of previous layer after applying activation function 
            i.e. g(z) or g(wx+b), Weight and bias of current layer, and current layer linear
            transformation i.e. z or wx+b
            (Cached during feed forward step)
        activation : str
            type of activation function used in hidden layers
        samples : int
            number of samples 
        lambd : float
            L2 regularization value to be applied
        L2_regularization : Boolean
            L2 regularization to be applied or not

        Returns
        -------
        derivative_preActivation : array
            array containing values representing derivative of loss function upto previous layer
        derivative_currentW : array
            array containing values representing derivative of loss function with respect to
            current layer weights
        derivative_currentB : array
            array containing values representing derivative of loss function with respect to
            current layer bias
        """
        cache_preActivation_currentW_currentB, cache_currentZ = cache_preActivation_currentW_currentB_currentZ
      
        #as explained in above function this dZ is derivative with respect to output of next layer
        #where output is taken after applying any activation function on linear combinations
        
        if activation == "relu":
            derivative_currentZ = self._relu_backward(derivative_postActivation, cache_currentZ)
              
        elif activation == "sigmoid":
            derivative_currentZ = self._sigmoid_backward(derivative_postActivation, cache_currentZ)
            # dA is derivative of loss upto this layer output
            # sigmoid_backward function internally outputs dA*(activation_cache*(1-activation_cache))
      
        derivative_preActivation, derivative_currentW, derivative_currentB = self._linear_backward(derivative_currentZ, cache_preActivation_currentW_currentB,samples,lambd,L2_regularization)
      
        return derivative_preActivation, derivative_currentW, derivative_currentB

    @staticmethod
    def _linear_backward(derivative_currentZ, cache_preActivation_currentW_currentB,samples,lambd,L2_regularization):
        """
        Given derivative values upto current layer linear transformation,
        It returns derivative values upto previous layer,
        It also returns derivative with repect to current layer weights and bias
        
        Parameters
        ----------
        derivative_currentZ : array
            array containing values representing derivative of loss upto
            current layer linear transformation 
        cache_preActivation_currentW_currentB : tuple
            tuple containing Output of previous layer after applying activation function 
            i.e. g(z) or g(wx+b), Weight and Bias of current layer
            (Cached during feed forward step)
        samples : int
            number of samples 
        lambd : float
            L2 regularization value to be applied
        L2_regularization : Boolean
            L2 regularization to be applied or not

        Returns
        -------
        derivative_preActivation : array
            array containing values representing derivative of loss function upto previous layer
        derivative_currentW : array
            array containing values representing derivative of loss function with respect to
            current layer weights
        derivative_currentB : array
            array containing values representing derivative of loss function with respect to
            current layer bias
        """        
        pre_Activation, current_W, current_b = cache_preActivation_currentW_currentB
        
        m = pre_Activation.shape[1] #we divide it by m because this currentZ/node is calculated by
        #taking weighted sum of all previous nodes/pre_activations.
        if L2_regularization==True:
            derivative_currentW = 1./m * np.dot(derivative_currentZ,pre_Activation.T) + ((lambd /samples)*current_W)
        else:
            derivative_currentW = 1./m * np.dot(derivative_currentZ,pre_Activation.T) + ((lambd /samples)*current_W)
        derivative_currentB = 1./m * np.sum(derivative_currentZ, axis = 1, keepdims = True)
        derivative_preActivation = np.dot(current_W.T, derivative_currentZ)
      
        assert (derivative_preActivation.shape == pre_Activation.shape)
        assert (derivative_currentW.shape == current_W.shape)
        assert (derivative_currentB.shape == current_b.shape)
      
        return derivative_preActivation, derivative_currentW, derivative_currentB

    def _L_model_backward(self,output, Y, LST_cache_preActivation_currentW_currentB_currentZ,dropout,keep_prob,
                         cache_dropout,samples,lambd,L2_regularization,internal_activation,output_activation):
        """
        This function generates gradient for each layer given actual, predicted,
        and all cache values.
        
        Parameters
        ----------
        output : array
            final predicted values after feed forward step
        Y : array
            actual output values
        LST_cache_preActivation_currentW_currentB_currentZ : list
            list of tuples containing array values for each layer. 
            For each layer each layer tuple inside list contains previous layer 
            output after applying activation function, current layer weights, bias, 
            and current layer linear transformation before activation function
            (cached during feed forward step)
        dropout : Boolean
            if True then dropout regularization is applied
        keep_prob : float
            amount of dropout regularization 
            (probability of keeping the nodes alive)
        cache_dropout : list
            list cotaining array for each layer.
            For each layer the information of which nodes were kept during random
            dropout regularization was cached in form of binary arrays having 1 or 0 values
            (cached during feed forward step)
        samples : int
            number of samples
        lambd : float
            L2 regularization value to be applied
        L2_regularization : Boolean
            L2 regularization to be applied or not
        internal_activation : str
            type of activation function for hidden layer
        output_activation : str
            type of activation function for output layer

        Returns
        -------
        grads : dict
            gradients for each layer inputs, weights and biass i.e.
            grads["dA1"] = derivative_preActivation
            grads["dW1"] = derivative_currentW
            grads["db1"] = derivative_currentB
            grads["dA2"] = derivative_preActivation
            grads["dW2"] = derivative_currentW
            grads["db2"] = derivative_currentB
            grads["dA3"] = derivative_preActivation
            grads["dW3"] = derivative_currentW
            grads["db3"] = derivative_currentB
            here 1,2,3, represent layer number
        """          
        grads = {}
        L = len(LST_cache_preActivation_currentW_currentB_currentZ) # the number of layers excluding input layer
        Y = Y.reshape(output.shape) 
      
        # Initializing the backpropagation
        derivative_output = - (np.divide(Y, output) - np.divide(1 - Y, 1 - output))
        #derivative with respect to loss for log loss function
      
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        derivative_postActivation  = derivative_output
        cache_preActivation_currentW_currentB_currentZ = LST_cache_preActivation_currentW_currentB_currentZ[-1] #This current_cache contains cache for final layer        
    
        derivative_preActivation, derivative_currentW, derivative_currentB = self._linear_activation_backward(derivative_postActivation,cache_preActivation_currentW_currentB_currentZ,output_activation,samples,lambd,L2_regularization)
        
        #cache_tmp is a list containing tmp array of 0,1 / on,off for second to second_last layer excluding input and output layer
        if dropout:
            derivative_preActivation = np.multiply(derivative_preActivation, cache_dropout[-1])  # Step 1: Apply mask cache_tmp to shut down the same neurons as during the forward propagation
            derivative_preActivation /= keep_prob                # Step 2: Scale the value of neurons that haven't been shut down
      
        grads["dA" + str(L)] = derivative_preActivation
        grads["dW" + str(L)] = derivative_currentW
        grads["db" + str(L)] = derivative_currentB
      
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            derivative_postActivation = grads["dA"+str(l+2)]
            cache_preActivation_currentW_currentB_currentZ = LST_cache_preActivation_currentW_currentB_currentZ[l]
            derivative_preActivation, derivative_currentW, derivative_currentB = self._linear_activation_backward(derivative_postActivation, cache_preActivation_currentW_currentB_currentZ,internal_activation,samples,lambd,L2_regularization)
            if dropout and l>0:
                #print(derivative_preActivation.shape,cache_tmp[l-1].shape)
                derivative_preActivation = np.multiply(derivative_preActivation, cache_dropout[l-1])  
                # Step 1: Apply mask cache_tmp to shut down the same neurons as during the forward propagation
                derivative_preActivation = derivative_preActivation/keep_prob
                                             
            grads["dA" + str(l + 1)] = derivative_preActivation
            grads["dW" + str(l + 1)] = derivative_currentW
            grads["db" + str(l + 1)] = derivative_currentB
        
        return grads
      
    #Implementation   
    @staticmethod
    def _update_parameters(parameters, grads, learning_rate,optimization,
                          alpha,eps, nu_W, nu_b, G_W, G_b):
        """
        Given parameters and gradients for those parameters,
        It returns updated parameters.

        Parameters
        ----------
        parameters : dict
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        grads : dict
            gradients for each layer inputs, weights and biass i.e.
            grads["dA1"] = derivative_preActivation
            grads["dW1"] = derivative_currentW
            grads["db1"] = derivative_currentB
            grads["dA2"] = derivative_preActivation
            grads["dW2"] = derivative_currentW
            grads["db2"] = derivative_currentB
            grads["dA3"] = derivative_preActivation
            grads["dW3"] = derivative_currentW
            grads["db3"] = derivative_currentB
            here 1,2,3, represent layer number
        learning_rate : float
            learning rate to be applied
        optimization : str
            type of optimization function to be used
        alpha : float
            alpha value in case optimization is other than gradient descent
        eps : float
            epsilon value in case optimization is other than gradient descent
        nu_W : float
            nu_W value in case optimization is other than gradient descent
        nu_b : float
            nu_b value in case optimization is other than gradient descent
        G_W : float
            G_W value in case optimization is other than gradient descent
        G_b : float
            G_b value in case optimization is other than gradient descent

        Returns
        -------
        parameters : dict 
            updated parameters 
        nu_W : float
            updated nu_W value in case optimization is other than gradient descent
        nu_b : float
            updated nu_b value in case optimization is other than gradient descent
        G_W : float
            updated G_W value in case optimization is other than gradient descent
        G_b : float
            updated G_b value in case optimization is other than gradient descent
        """
        L = len(parameters) // 2 # number of layers in the neural network
        
        # Update rule for each parameter. Use a for loop.
        for l in range(L):            
            if optimization == "GD":
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
          
            elif optimization == "GD_Momentum":
                #nu_W = (alpha * nu_W) + (learning_rate * grads["dW" + str(l+1)])
                #nu_b = (alpha * nu_b) + (learning_rate * grads["db" + str(l+1)])
                #parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - nu_W
                #parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - nu_b

                print ("Currently, there is an issue with GD_Momentum")
                print ()
                print ("STOP AND USE, optimization = 'GD' instead")
                
            elif optimization == "RMSprop":
                #G_W = (alpha*G_W)+((1-alpha)*(grads["dW" + str(l+1)]**2))
                #G_b = (alpha*G_b)+((1-alpha)*(grads["db" + str(l+1)]**2))
                #parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - ((learning_rate/np.sqrt(G_W+eps))*grads["dW" + str(l+1)])
                #parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - ((learning_rate/np.sqrt(G_b+eps))*grads["db" + str(l+1)])
                
                print ("Currently, there is an issue with RMSprop")
                print ()
                print ("STOP AND USE, optimization = 'GD' instead")
                
        return parameters, nu_W, nu_b, G_W, G_b
    
    def fit(self,X,Y):
        """
        Given input and output along with different options set and defined during
        making of class object,
        This function returns optimized weight and bias parameters for neural network.

        Parameters
        ----------
        X : array
            input array of shape (input features, samples)
        Y : array
            output array of shape (output features, samples)

        Returns
        -------
        parameters : dict
            updated parameters
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...
        """
        num_iterations=self.num_iterations
        learning_rate= self.learning_rate
        layers_dims = self.layer_dims
        weight_initialization = self.weight_initialization
        seed = self.seed
        internal_activation= self.internal_activation
        output_activation = self.output_activation 
        L2_regularization = self.L2_regularization
        dropout = self.dropout
        lambd = self.lambd
        keep_prob = self.keep_prob
        print_cost = self.verbose 
        optimum_parameters = self.optimum_parameters
        optimization = self.optimization
        alpha = self.alpha
        eps = self.eps
        
        samples = X.shape[1]
        grads = {}
        costs = [] # to keep track of the loss
        #m = X.shape[1] # number of examples
    
        if optimum_parameters is not None:
            parameters = optimum_parameters
        else:
            parameters = self._initialize_parameters_deep(layers_dims,weight_initialization,seed)
              
        nu_W = 0
        nu_b = 0
        G_W = 0
        G_b = 0 
                          
        # Loop (gradient descent)
        for i in range(0, num_iterations):                
            # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
            output, LST_cache_preActivation_currentW_currentB_currentZ, cache_dropout = self._L_model_forward(X, 
                   parameters,dropout,keep_prob, internal_activation,output_activation)                
            # Loss
            cost = self._cross_entropy_cost(output, Y, parameters, lambd, L2_regularization)  
            # Backward propagation.
            grads = self._L_model_backward(output, Y, LST_cache_preActivation_currentW_currentB_currentZ,
                                     dropout,keep_prob,cache_dropout,samples,lambd,L2_regularization,
                                     internal_activation,output_activation)                  
            # Update parameters.
            parameters, nu_W, nu_b, G_W, G_b = self._update_parameters(parameters, grads, learning_rate,optimization,
                                                                      alpha,eps, nu_W, nu_b, G_W, G_b)
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                    
        # plot the loss
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
      
    def predict(self,X, parameters):
        """
        Given training data input and optimized parameters,
        This function returns predicted value
        
        Parameters
        ----------
        X : array
            input array of shape (input features, samples)
        parameters : dict
            updated parameters
            dict keys will be W1,b1,W2,b2,W3,b3... and values will be numpy array. 
            Here 1,2,3 are hidden layer number.
            Shape of numpy array for W1 will be (hidden layer1, input layer)
            Shape of numpy array for b1 will be (hidden layer1, 1)
            Shape of numpy array for W2 will be (hidden layer2, hidden layer1)
            Shape of numpy array for b2 will be (hidden layer2, 1)
            Shape of numpy array for W3 will be (hidden layer3, hidden layer2)
            Shape of numpy array for b3 will be (hidden layer3, 1)
            and so on...

        Returns
        -------
        p : array
            final prediction values using optimized parameters
        """          
        internal_activation=self.internal_activation
        output_activation=self.output_activation
        
        m = X.shape[1]
        #n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
      
        # Forward propagation
        dropout = False
        keep_prob = None
        probas, _, _ = self._L_model_forward(X, parameters,dropout,keep_prob,internal_activation,output_activation)
              
        # convert probas to 0/1 predictions
        
        p = np.where(probas>=0.5,1,0)
        
        #for i in range(0, probas.shape[1]): 
         #     if probas[0,i] > 0.5:
          #          p[0,i] = 1
           #   else:
            #        p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))  
        return p

