import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import fast
import Trainer
from visu import draw_encoding, draw_weights
from Manager import save_weights, load_weights


def main():


    # user input (ideally)
    df_name = "mnist"
    #df_name = "fashion_mnist"
    n_hidden = 100

    # load data
    file_name = f'{df_name}.csv'
    relative_path =   '/../data'
    save_path = os.path.dirname(__file__) +  relative_path
    full_path = os.path.join(save_path, file_name)
    assert  os.path.exists(full_path), f'File {file_name} not found'
    data = np.loadtxt(full_path , delimiter=",")
    
    # preprocessing    
    imgs = np.asfarray(data[:, 1:])
    imgs = imgs/255.0 # normalize 
    labels = np.asfarray(data[:, :1])

    fig=plt.figure(figsize=(12.9,10))
   # draw_weights(imgs,10,10,fig, save=True, df_name=df_name) # its the right matrix

    # transform labels into one hot representation
    n_labels = len(np.unique(labels)) 
    label_list = np.arange(n_labels)
    labels_onehot = (label_list==labels).astype(np.float64)

 
    # train unsupervised layer
    # weights = fast.training(imgs, n_hidden)
    # fig=plt.figure(figsize=(12.9,10))
    # draw_weights(weights,10,10,fig,epoch=99, save=True, df_name=df_name) # its the right matrix
    # save_weights(weights,"unsupervised", df_name, notes="Epoch100")

    weights = load_weights("unsupervised", df_name, notes="Epoch200")
    
    # "encoding" all the inputs by running them through unsupervised layer
    encoding = np.dot(weights,imgs.T)
    encoding = np.where(encoding<0, 0, encoding) # RELU at home

    fig=plt.figure(figsize=(12.9,10))
    draw_encoding(encoding, 10, 10, fig, df_name)

    # train supervised trainer 
    train = Trainer.Trainer()
    train.training_loop(encoding.T, labels_onehot)
    train.visualize_training(df_name)

    return 0



if __name__ == "__main__":
    main()