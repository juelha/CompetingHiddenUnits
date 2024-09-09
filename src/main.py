import numpy as np
import matplotlib.pyplot as plt
import os
from BioLearning import biolearning
import Trainer
from visu import draw_encoding, draw_weights, monitoring
from Manager import load_data,save_weights, load_weights, save_encoding

def main():
    df_name = "mnist"
    inputs, labels = load_data(df_name)
    
    # train unsupervised layer
    n_hidden = 100
    n_epochs = 200
   # weights = biolearning(inputs, n_hidden, n_epochs, df_name)
   # save_weights(weights, df_name, notes=f"Epoch{n_epochs}_{n_hidden}hidden")
    weights = load_weights(df_name, notes=f"Epoch{n_epochs}_{n_hidden}hidden")

    draw_weights(weights, 10, 10, df_name, epoch=n_epochs, n_hidden=n_hidden, save=True) # its the right matrix
    

    # "encoding" all the inputs by running them through unsupervised layer
    encoding = np.dot(weights, inputs.T)
    #encoding = np.where(encoding<0, 0, encoding) # RELU <- performance seems better without 
    encoding /= np.max(np.absolute(encoding)) 

    encoding = encoding.T # make encoding have shape (n_samples, n_hidden)
    
    draw_encoding(encoding, 10, 10, df_name, save=True)
    save_encoding(encoding, df_name)
    
    monitoring(weights, inputs, encoding, df_name)


    # train supervised trainer 
    train = Trainer.Trainer(n_inputs = n_hidden, n_outputs = labels.shape[1])
    train.training_loop(encoding, labels)
    train.visualize_accuracy(df_name)
    train.visualize_error(df_name)

    return 0



if __name__ == "__main__":
    main()