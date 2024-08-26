# basics 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets


from tqdm import tqdm 


class Trainer():
    """Training the model while keeping track of the accuracies and losses using SGD.
    ___________________________________________________________
    """

    def __init__(self, optimizer_func="Adam", learning_rate=None, n_epochs=None):
        """Initializes Trainer()-Object
        
        Args:
            arc (MLP): architecture of the NN
            n_epochs (int): iterations of training loop
            learning_rate (float): learning rate for training
            optimizer_func (tf.keras.optimizers)
        
        Attributes:
            loss_func
            test_accuracies (list(float)): keeping track of test accuracies during training
            test_losses (list(float)): keeping track of test losses during training
            train_losses (list(float)): keeping track of train losses during training
        """

        # hyperparamers
        self.n_epochs = 10
        self.learning_rate = 1.0
        self.batch_size = 100
        mu=0.0       # the mean of the gaussian distribution that initializes the weights
        sigma=1.0    # the standard deviation of that gaussian
        self.weights  = np.random.normal(mu, sigma, (100,10)) # init weights 
        #self.optimizer_func = optimizer_func(learning_rate)

        # get params
        self.params = [
            self.n_epochs, self.learning_rate,
        ]
        
        # for visualization of training
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []

        self.v_accuracies = []
        self.v_losses = []


    def __call__(self, train_ds,  test_ds, validation_ds,  constraint_center, constraint_width, learning_rate, n_epochs):
        """Calling the trainer calls training_loop()
        Args: 
            train_ds (tuple of two numpy.ndarrays): dataset for training
            test_ds (tuple of two numpy.ndarrays): dataset for testing
        Note: 
            Implemented for cleaner code and the use in the inherited 
            class neurofuzzyTrainer() 
        """
        self.training_loop(train_ds,  test_ds, validation_ds,  constraint_center, constraint_width, learning_rate, n_epochs)



    def training_loop(self, imgs, labels, validation_ds_og=None, n_epochs=None):
        """Training of the model
        Args: 
            train_ds (tuple of two numpy.ndarrays): dataset for training (input, target)
            test_ds (tuple of two numpy.ndarrays): dataset for testing (input, target)
        """

        x_train, x_test = np.split(imgs,[int(0.8 * len(imgs))])
        y_train, y_test = np.split(labels,[int(0.8 * len(labels))])
        #print(x_train.shape)
        
        # training loop until self.n_epochs
        for epoch in range(self.n_epochs):

            # do the shuffle 
            shuffled_indices_tr = np.random.permutation(x_train.shape[0]) # return a permutation of the indices
            shuffled_indices_te = np.random.permutation(x_test.shape[0]) # return a permutation of the indices
            x_train, x_test = x_train[shuffled_indices_tr,:], x_test[shuffled_indices_te,:]
            y_train, y_test = y_train[shuffled_indices_tr,:], y_test[shuffled_indices_te,:]

            # iterate over all minibatches <- MAKE PRETTIER
            for i in range(imgs.shape[1]//self.batch_size):
                x_train_batch = x_train[i*self.batch_size:(i+1)*self.batch_size,:] # pick minibatch, has shape (batch_size, n_hidden)
                x_test_batch = x_test[i*self.batch_size:(i+1)*self.batch_size,:] 
                y_train_batch = y_train[i*self.batch_size:(i+1)*self.batch_size,:] 
                y_test_batch = y_test[i*self.batch_size:(i+1)*self.batch_size,:] 


                # run model on test_ds to keep track of progress during training
                self.test(x_test_batch, y_test_batch)

                # train 
                self.train_step(x_train_batch, y_train_batch)

                print(f'Epoch: {str(epoch)} with \n \
                test accuracy {self.test_accuracies[-1]} \n \
                train accuracy {self.train_accuracies[-1]} \n \
                test loss {self.test_losses[-1]} \n \
                train loss {self.train_losses[-1]}')
                

        print("Training Loop completed")
        return 0



    def test(self, x_batch, y_batch):
        """Forward pass of test_data
        Args:
            test_ds (TensorSliceDataset): batch of testing dataset
        Returns:
            test_loss (float): average loss from output to target
            test_accuracy (float): average accuracy of output
        """

        accuracy_aggregator = []
        loss_aggregator = []
        
       # for img, target in (zip(tqdm(x_batch, desc='testing'), y_batch)):
        z = self.z(x_batch, self.weights)
        prediction = self.act_func(z)
        loss = self.error_function(prediction, y_batch)
        accuracy =  y_batch == np.round(prediction, 0)

        # return averages per batch
        test_loss = np.mean(loss)
        test_accuracy =  np.mean(accuracy)

        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)
       # return test_loss, test_accuracy


    def train_step(self, x_train, y_train):
        """Implements train step for batch of datasamples
        Args:
            input (tf.Tensor): input sequence of a batch of dataset
            target (tf.Tensor): output sequence of a batch of dataset
        Returns:
            loss (float): average loss before after train step
        """
        losses_aggregator = []
        accuracy_aggregator = []


        # forward pass 
        z = self.z(x_train, self.weights)
        prediction = self.act_func(z)
        # print("pred", prediction)
        # get loss
        #print("tar", target)
        loss = self.error_function(prediction, y_train)
        accuracy =  y_train == np.round(prediction, 0)
        accuracy = np.mean(accuracy)
        losses_aggregator.append(loss)
        accuracy_aggregator.append(accuracy)

        # get gradients
        gradients = self.error_function_derived(prediction, y_train) * self.act_func_derived(z)
        gradients = np.dot(gradients.T, x_train).T

        # adapt the trainable variables with gradients 
        self.weights -= self.learning_rate * gradients

        # return average loss
        loss = np.mean(losses_aggregator)
        acc_mean = np.mean(accuracy_aggregator)
        self.train_accuracies.append(acc_mean)
        self.train_losses.append(loss)
        #return loss, acc_mean

    def z(self, x, weights):
        return np.dot(x.T,weights)
    
    def act_func(self,z):
        return  1/(1+np.exp(-z))
    
    def act_func_derived(self,z):
        f = self.act_func(z)
        df = f * (1 - f)
        return df

    def error_function(self, prediction, targets):
        """Derived error function:  
            error function: tf.reduce_mean(0.5*(prediction - targets)**2)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """

        error_term = 0.5*(prediction - targets)**2
        return error_term


    def error_function_derived(self, prediction, targets):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        error_term = prediction - targets
        return error_term


    def visualize_training(self, df_name="mnist", type_model="SGD"):
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
        save_path = os.path.dirname(__file__) +  f'/../reports/figures'
        completeName = os.path.join(save_path, file_name)
        plt.savefig(completeName)
        plt.clf()

    def summary(self):
        column_names = ['n_epochs', 'learning_rate', 'loss_func', 'optimizer_func']
        d = self.params
        df = pd.DataFrame(d, column_names)
        return df
    

###############################################################################################################

# dataset = datasets.load_iris()
# data = dataset['data']
# labels = dataset['target']
# var_names = dataset['feature_names']
# target_names = dataset['target_names']
# print(var_names)
# print(np.unique(labels))
# print(target_names)


# # transform labels into one hot representation
# lr = np.arange(len(np.unique(labels)))
# labels = np.reshape(labels, (labels.shape[0],1))
# labels_onehot = (lr==labels).astype(np.float64)
# print(labels_onehot)


# tr = Trainer()
# tr.training_loop(imgs=data, labels=labels_onehot)
