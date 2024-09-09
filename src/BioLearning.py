import numpy as np
import matplotlib.pyplot as plt
#from visu import draw_weights, draw_encoding
from src import *
from tqdm import tqdm 

def biolearning(inputs, n_hidden, n_epochs, df_name, show=False): 
    """Fast implementation of biological learning algorithm, 
    rewritten from https://github.com/DimaKrotov/Biological_Learning/tree/master (c) 2018 Dmitry Krotov -- Apache 2.0 License 

    Args:
        inputs (np.ndarray): input matrix of shape (n_samples, n_features)
        n_hidden (int): number of hidden units
        n_epochs (int): number of max epochs

    Returns:
        weights (np.ndarray): weight matrix of shape (n_samples, n_features)
    """

    
    n_samples, n_pixel = inputs.shape

    # hyperparameters
    eps0=2e-2    # learning rate
    mu=0.0       # the mean of the gaussian distribution that initializes the weights
    sigma=1.0    # the standard deviation of that gaussian
    size_batch=100      # size of the minibatch
    prec=1e-30   # parameter that controls numerical precision of the weight updates
    delta=0.4    # Strength of the anti-hebbian learning
    p=2.0        # Lebesgue norm of the weights
    k=2          # ranking parameter, must be integer that is bigger or equal than 2

    fig=plt.figure(figsize=(12.9,10))

    weights = np.random.normal(mu, sigma, (n_hidden, n_pixel)) # init weights 

    # training loop
    for epoch in tqdm(range(n_epochs), desc="bio-learning"):
        eps = eps0 * (1-epoch/n_epochs) # annealing the learning rate 
        inputs = inputs[np.random.permutation(n_samples),:] # shuffle

        # iterate over all minibatches 
        for i in range(n_samples//size_batch):
            minibatch = np.transpose(inputs[i*size_batch:(i+1)*size_batch,:]) # pick minibatch, has shape (n_features, size_batch)

            # why split sign only to mutliply it in the end anyway?
            sig = np.sign(weights) 
            input_currents = np.dot(sig*np.absolute(weights)**(p-1),minibatch) # convert raw input into set of input currents Iμ [eq. 8]
                                                                               # has shape (n_hidden, size_batch)
            
            ranking = np.argsort(input_currents, axis=0) # use currents as a proxy for ranking of the final activities
           
            # [eq. 10]
            g_h = np.zeros((n_hidden, size_batch))
            g_h[ranking[n_hidden-1,:],np.arange(size_batch)] = 1.0 # winner takes all
            g_h[ranking[n_hidden-k],np.arange(size_batch)] = -delta # inhibitory effect
            
            # plasticity rule 
            # g_h is multiplied out to both sides of the minus <- why?
            xx = np.sum(np.multiply(g_h, input_currents), axis=1) # sum (g(Q) * {W,v} )  
            xx = xx.reshape(xx.shape[0], 1) # (100, 1)                             
            xx_ = np.tile(xx, (1,n_pixel) ) # (100, 784)

            r_of_minus = np.multiply(xx_, weights) # xx_shape == weights.shape = (100, 784)

            # right hand side of [3]
            delta_w = np.dot(g_h, np.transpose(minibatch)) - r_of_minus   

            # scaling down weight changes by their max
            max_delta_w = np.amax(np.absolute(delta_w)) 
            if max_delta_w<prec:
                max_delta_w=prec 

            # adapt weights
            weights += eps*np.true_divide(delta_w, max_delta_w) 

        draw_weights(weights, 10, 10, df_name, fig, epoch+1, n_hidden, show)
      

    return weights
            