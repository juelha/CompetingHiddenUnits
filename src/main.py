import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import fast
import Trainer
from visu import draw_encoding, draw_weights
from Manager import save_weights, load_weights

def f(x, n=1):

    for i in range(x.shape[0]):
        if x[i,:]>=0: 
            return x[i,:]**n
        else:
            return 0

def main():

    # # load data
    # file_name = 'mnist_all.mat'
    # relative_path =   '/../data'
    # save_path = os.path.dirname(__file__) +  relative_path
    # full_path = os.path.join(save_path, file_name)
    # assert  os.path.exists(full_path), f'File {file_name} not found'
    # # read scipy.io
    # mat = scipy.io.loadmat(full_path)


    
    # # ideally get this from data 
    # n_class_units=10 
    # n_pixel=784  # per img
    # n_examples=60000

    # # preprocessing 
    # inputs=np.zeros((0,n_pixel)) # empty array in correct shape for later (unneccessary??)
    # for i in range(n_class_units): # iterate over mat dictionary 
    #     inputs=np.concatenate((inputs, mat['train'+str(i)]), axis=0) # get entries of dict 
    # inputs=inputs/255.0 # normalize 
        
    
    # fast.training(inputs, n_pixel, n_examples)

    df_name = "mnist"

    # load data
    
    file_name = 'mnist.csv'
    relative_path =   '/../data'
    save_path = os.path.dirname(__file__) +  relative_path
    full_path = os.path.join(save_path, file_name)
    assert  os.path.exists(full_path), f'File {file_name} not found'


    # preprocessing

    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "../data/"
    data = np.loadtxt(full_path , delimiter=",")

    
    

    print(data[:10])
    imgs = np.asfarray(data[:, 1:])
    print(imgs[:10])
    imgs=imgs/255.0 # normalize 

    fig=plt.figure(figsize=(12.9,10))
    draw_weights(imgs,10,10,fig,50)

    print(imgs[:10])
    labels = np.asfarray(data[:, :1])
    print(labels[:10])


    # transform labels into one hot representation
    lr = np.arange(no_of_different_labels)
    labels_onehot = (lr==labels).astype(np.float64)
    print(labels_onehot[:10])

    # for i in range(10):
    #     img = imgs[i].reshape((28,28))
    #     plt.imshow(img, cmap="Greys")
    #     plt.show()

    n_class_units=10 
    n_pixel=784  # per img
    n_examples=60000
    n_hidden = 100
    size_batch =100
   # fig=plt.figure(figsize=(12.9,10))

    #weights = fast.training(imgs, n_hidden)
    #draw_weights(weights,10,10,fig,epoch=199, save=True) # its the right matrix

#     save_weights(weights,"unsupervised", df_name, notes="200epochs")


    weights = load_weights("unsupervised", df_name, notes="200epochs")
    fig=plt.figure(figsize=(12.9,10))
   # draw_weights(weights,10,10,fig,199)
    #print(np.array_equal(weights,w)) # check if saving works <- True!


    # "encoding" all the inputs by running them through unsupervised layer
   # encoding = np.zeros((n_examples, size_batch))
  #  encoding = np.zeros((n_examples, ))
    print("img", imgs.shape)

    # for i in range(n_examples//size_batch):
    #     inputs = imgs[i*size_batch:(i+1)*size_batch,:] # pick minibatch 
    #     #print("\nin", inputs.shape)
    #    # print("\ndot", ( np.dot(weights**(2-1),inputs)).shape)
    #    # print("en", encoding)
    #   #  print("en", encoding[0].shape)
    #     encoding[i*size_batch:(i+1)*size_batch,:] = np.dot(weights**(2-1),np.transpose(inputs))
    #    # encoding[i*size_batch:(i+1)*size_batch,:] = np.where(encoding[i*size_batch:(i+1)*size_batch,:]<0, 0, encoding[i*size_batch:(i+1)*size_batch,:] ) # RELU?


    #encoding = np.zeros((n_examples, size_batch))
    #encoding = np.dot(weights**(2-1),np.transpose(imgs))
    encoding = np.dot(weights,imgs.T)
    encoding = np.where(encoding<0, 0, encoding) # RELU at home
    # for i in range(n_examples):
    #     idk = np.dot(weights**(2-1),np.transpose(imgs[i,:]))
    #     print(idk.shape)
    #     idk = np.where(idk<0, 0, idk) # RELU at home
    #     encoding[i,:] = idk 
        
    fig=plt.figure(figsize=(12.9,10))
    print("en", encoding.shape)
    draw_encoding(encoding,20,20,fig)
    
    print(np.count_nonzero(encoding))
   # return 0
    train = Trainer.Trainer()
    train.training_loop(encoding.T, labels_onehot)
    train.visualize_training()

    return 0



if __name__ == "__main__":
    main()