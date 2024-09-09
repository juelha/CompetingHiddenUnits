# basics 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm 
import pandas as pd



class Trainer():
    """Training with SGD while keeping track of the accuracies and losses
    """

    def __init__(self, n_inputs, n_outputs, optimizer_func="Adam", learning_rate=None, n_epochs=None):
        """_summary_

        Args:
            n_inputs (_type_): _description_
            n_outputs (_type_): _description_
            optimizer_func (_type_, optional): _description_. Defaults to "Adam".
            learning_rate (_type_, optional): _description_. Defaults to None.
            n_epochs (_type_, optional): _description_. Defaults to None.

        Attributes:
            loss_func
            test_accuracies (list(float)): keeping track of test accuracies during training
            test_losses (list(float)): keeping track of test losses during training
            train_losses (list(float)): keeping track of train losses during training    
        """


        # hyperparamers
        self.n_epochs = 40
        self.learning_rate =  0.001
        self.batch_size = 100
        mu=0.0       # the mean of the gaussian distribution that initializes the weights
        sigma=1.0    # the standard deviation of that gaussian
        self.weights  = np.random.normal(mu, sigma, (n_inputs, n_outputs)) # init weights 
        
        # for visualization of training
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []

        self.v_accuracies = []
        self.v_losses = []


    def training_loop(self, imgs, labels):
        """_summary_

        Args:
            imgs (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        

        x_train, x_test = np.split(imgs,[int(0.8 * len(imgs))])
        y_train, y_test = np.split(labels,[int(0.8 * len(labels))])
        
        for epoch in range(self.n_epochs):

            print(x_train)
            # do the shuffle 
            shuffled_indices_tr = np.random.permutation(x_train.shape[0]) # return a permutation of the indices
            shuffled_indices_te = np.random.permutation(x_test.shape[0]) # return a permutation of the indices
            x_train, x_test = x_train[shuffled_indices_tr,:], x_test[shuffled_indices_te,:]
            y_train, y_test = y_train[shuffled_indices_tr,:], y_test[shuffled_indices_te,:]


            # iterate over all minibatches <- MAKE PRETTIER
           # for i in range(y_train.shape[0]//self.batch_size):

           # iterate over all examples 
          #  for i in range(y_train.shape[0]):

           # pick one minibatch per epoch
            x_train_batch = x_train[0:self.batch_size,:] # pick minibatch, has shape (batch_size, n_hidden)
            x_test_batch = x_test[0:self.batch_size,:] 
            y_train_batch = y_train[0:self.batch_size,:] 
            y_test_batch = y_test[0:self.batch_size,:] 
          #  y_test_batch = y_test[i*self.batch_size:(i+1)*self.batch_size,:] 

            # run model on test_ds to keep track of progress during training
            self.test(x_test_batch, y_test_batch)

            # train 
            self.train_step(x_train_batch, y_train_batch)

            print(f'Epoch: {str(epoch+1)} with \n \
            test accuracy {self.test_accuracies[-1]} \n \
            train accuracy {self.train_accuracies[-1]} \n \
            test loss {self.test_losses[-1]} \n \
            train loss {self.train_losses[-1]}')
                

        print("Training Loop completed")
        return 0



    def test(self, x_batch, y_batch):
        """_summary_

        Args:
            x_batch (_type_): _description_
            y_batch (_type_): _description_
        """

        # forward pass 
        z = self.z(x_batch, self.weights)
        prediction = self.tanh(z)
        loss = self.error_function(prediction, y_batch)
        accuracy = self.accuracy_func(prediction, y_batch)

        # save averages per batch
        self.test_losses.append(np.mean(loss))
        self.test_accuracies.append(np.mean(accuracy))


    def train_step(self, x_batch, y_batch):
        """_summary_

        Args:
            x_batch (_type_): _description_
            y_batch (_type_): _description_
        """

        # forward pass 
        z = self.z(x_batch, self.weights)
        prediction = self.tanh(z)
        loss = self.error_function(prediction, y_batch)
        accuracy = self.accuracy_func(prediction, y_batch)

        # save averages per batch
        self.train_losses.append(np.mean(loss))
        self.train_accuracies.append(np.mean(accuracy))
        
        # get gradients
        g = self.error_function_derived(prediction, y_batch) * self.tanh_der(z) #100,10
        gradients = np.dot(x_batch.T, g) #/ x_batch.shape[0] # ? 

        # adapt the trainable variables with gradients 
        self.weights -= self.learning_rate * gradients

        

    def z(self, x, weights):
        return np.dot(x,weights)
    
    def tanh(self,z):
        return np.tanh(z)
       # return (np.exp(z)  - np.exp(-z) )/(np.exp(z)  + np.exp(-z) )
        
    def tanh_der(self,z):
        return 1 - np.tanh(z)**2

    def error_function(self, prediction, targets, m=2):

        error_term = 1/m*(prediction - targets)**m       

        # calc percentual error
        max_error = 1/m*((-1*np.ones_like(prediction)) - targets)**m
        percentual = 100/max_error *  error_term
        percentual = np.mean(percentual, axis=1) # avg over units
        return percentual 


    def error_function_derived(self, prediction, targets, m=2):

        print("\nü", prediction[10])
        print("ä",targets[10])
        print("huh", (prediction - targets)[10])
        error_term = prediction - targets
        return error_term
      

    def accuracy_func(self, prediction, targets):
        """accuracy defined as number of correct predictions in %

        Args:
            prediction (_type_): _description_
            targets (_type_): _description_
        """
        
        pred_idx = np.argmax(prediction, axis=1) 
        tar_idx = np.argmax(targets, axis=1) 
        acc = np.where(pred_idx == tar_idx, 1, 0)
        print("\n\npred", prediction)
        print("\n\npred_idx", pred_idx)
        print("tar_idx", tar_idx)
        print("acc", acc)
        return acc

    def visualize_training(self, df_name, type_model="SGD"):
        """ Visualize accuracy and loss for training and test data.
        Args:
            type_model (str): type of model that has been trained
        """
        plt.figure()
        # accuracies
        line1, = plt.plot(self.test_accuracies)
        line2, = plt.plot(self.train_accuracies)
       # line3, = plt.plot(self.val_accuracies)
        # losses
        line4, = plt.plot(self.test_losses)
        line5, = plt.plot(self.train_losses)
       # line5, = plt.plot(self.v_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1, line2, line4, line5),
        ("test accuracy", "training accuracy", "test losses", "training losses"))
        plt.title(f'{type_model}')
        plt.figure
       
        # get save path 
        file_name = 'Performance'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures'
        completeName = os.path.join(save_path, file_name)
        plt.savefig(completeName)
        plt.clf()
