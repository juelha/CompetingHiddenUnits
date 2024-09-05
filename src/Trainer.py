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

    def __init__(self, n_inputs, n_outputs, optimizer_func="Adam", learning_rate=None, n_epochs=None):
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
        self.n_epochs = 50
        self.learning_rate =  0.001
        self.batch_size = 100
        mu=0.0       # the mean of the gaussian distribution that initializes the weights
        sigma=1.0    # the standard deviation of that gaussian
        self.weights  = np.random.normal(mu, sigma, (n_inputs, n_outputs)) # init weights 
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
           # for i in range(y_train.shape[0]//self.batch_size):

           # iterate over all examples 
          #  for i in range(y_train.shape[0]):

           # pick one minibatch per epoch
            x_train_batch = x_train#[0:self.batch_size,:] # pick minibatch, has shape (batch_size, n_hidden)
            x_test_batch = x_test#[0:self.batch_size,:] 
            y_train_batch = y_train#[0:self.batch_size,:] 
            y_test_batch = y_test#[0:self.batch_size,:] 
          #  y_test_batch = y_test[i*self.batch_size:(i+1)*self.batch_size,:] 

            print("xdx", y_test_batch)


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
        print("Xsha",x_batch)
        print("Xsha",x_batch.shape)
       # print("Xsha",x_batch[10])
      #  print("self.weights", self.weights[10])
        z = self.z(x_batch, self.weights)
       # print("z", z[10])
        prediction = self.tanh(z)
        loss = self.error_function(prediction, y_batch)
        accuracy =  self.accuracy_func(prediction, y_batch)

        # return averages per batch
        #test_loss = np.mean(loss)
        test_accuracy =  np.mean(accuracy)

        self.test_losses.append(loss)
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
        prediction = self.tanh(z)
        # print("pred", prediction)
        # get loss
        #print("tar", target)
        loss = self.error_function(prediction, y_train)
        accuracy =  y_train == np.round(prediction, 0)
        accuracy = np.mean(accuracy)
        losses_aggregator.append(loss)
        accuracy_aggregator.append(accuracy)

        # get gradients
        h = self.error_function_derived(prediction, y_train)
        print("sh", h)
        gradients = self.error_function_derived(prediction, y_train) * self.tanh_der(z) #100,10
        print("\ngr", gradients.shape)
        gradients = np.dot(x_train.T,gradients ) # ? 
        print("\ngr", gradients.shape)

        # adapt the trainable variables with gradients 
        self.weights -= self.learning_rate * gradients

        # return average loss
        #loss = np.mean(losses_aggregator) 
        acc_mean = np.mean(accuracy_aggregator)
        self.train_accuracies.append(acc_mean)
        self.train_losses.append(loss)
        #return loss, acc_mean

    def z(self, x, weights):
        return np.dot(x,weights)
    
    def tanh(self,z):
        return np.tanh(z)
       # return (np.exp(z)  - np.exp(-z) )/(np.exp(z)  + np.exp(-z) )
    
    def cosh(self,z):
        return(np.exp(z)  + np.exp(-z) )/2
        
    def tanh_der(self,z):
        return 1 - np.tanh(z)**2
       # return 1/np.cosh(z)**2
      #  return 1/self.cosh(z)**2

    
    def sigmoid(self,z):
        return  1/(1+np.exp(-z))
    
    def sigmoid_derived(self,z):
        f = self.sigmoid(z)
        df = f * (1 - f)
        return df


    def cross_entropy_loss_prime(self, p,t):
    # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
        return  np.sum(p- t, axis=1) / len(t)


    def cross_entropy_loss(self, p, t):
        return - np.sum(t * np.log(p), axis=1) / len(t)
 

    def error_function(self, prediction, targets, m=2):
        """Derived error function:  
            error function: tf.reduce_mean(0.5*(prediction - targets)**2)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """

        error_term = (prediction - targets)**m
       # print("\nprediction", prediction)
        
        print("\nt", targets.shape)
        print("\ne", error_term.shape)
        print("\n\n\n HERE HERE HERE")
     #   print("\n\nü", prediction[10])
     #   print("\nä",targets[10])
     #   print("\nö",error_term[10])
        
        
        # print("\n\nü", prediction[10])
        # print("\nä",targets[10])
        # print("\nö",error_term[10])
        # error_term = np.sum(error_term, axis=1) * 1/10# sum over units
        # print("\ne after", error_term.shape)
        # print("\nö after",error_term[10])
        # print("boing", targets.shape[0])
        # error_term = np.sum(error_term, axis=1) # sum over units
        # return np.sum(error_term, axis=0)  # sum over examples and avg over examples
    
        error_term = np.mean(error_term,axis=1)
        return np.mean(error_term, axis=0)


    def error_function_derived(self, prediction, targets, m=2):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        print("\nü", prediction[10])
        print("\nä",targets[10])
        print("huh", (prediction - targets)[10])
        error_term = m*(prediction - targets)**(m-1)
        print("hah", error_term[10])
       # return np.mean(error_term, axis=1)
       # return error_term
      #  error_term = np.mean(error_term,axis=1)
      #  return np.mean(error_term, axis=0)
       # return error_term
        error_term = np.sum(error_term, axis=1)# * 1/10 # sum over units
        print("hahah", error_term[10])
        e = np.sum(error_term, axis=0)
        print("hahahah", e)
        return e #* 1/targets.shape[0] # sum over examples 

    def accuracy_func(self, prediction, targets):
        acc = (np.round(prediction, 0) == targets)
        # print("\n\n\n HERE HERE HERE")
        # print("\n\nü", prediction[10])
        # print("\nä",targets[10])
       #print("\nö",acc[10])
        acc = np.sum(acc, axis=1)
       # print("\nö after",acc[10])
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


# tr = Trainer(n_inputs=4, n_outputs=3)
# tr.training_loop(imgs=data, labels=labels_onehot)
# tr.visualize_training(df_name='mnist')