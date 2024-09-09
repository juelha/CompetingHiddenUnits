import numpy as np
import os 
import yaml 
from yaml.loader import UnsafeLoader
import pandas as pd

"""
Collection of functions to save and load
"""


def get_path(file_name, relative_path, notes=None):
    """gets full path for loading

    Args:
        file_name (str): name of file
        relative_path (str): path taken from file to source of data
        notes (str, optional): Defaults to None.

    Returns:
        full_path (str): full path for loading directly
    """
    save_path = os.getcwd() + relative_path # https://stackoverflow.com/questions/39125532/file-does-not-exist-in-jupyter-notebook
    #save_path = os.path.dirname(__file__) +  relative_path
    if not notes == None:
        file_name += "_" + notes
    full_path = os.path.join(save_path, file_name)
    return full_path


def load_data(df_name):
    """loads given dataset and splits into inputs and one-hot encoded labels

    Args:
        df_name (str): name of dataset

    Returns:
        inputs, labels_onehot (np.ndarray, np.ndarray): inptus of shape (n_samples, n_features), 
                                                        labels_onehot (n_samples, n_classes)
    """
    # load data
    file_name = f'{df_name}.pkl'
    relative_path =   '/data'
    path = get_path(file_name, relative_path)
    #data = np.loadtxt(path , delimiter=",")
    data = pd.read_pickle(path) 
    data = data.to_numpy()

    if df_name == "xor":
        labels = np.asfarray(data[:, -1:]) # labels are last column
        inputs =  np.asfarray(data[:, :-1])
    
    if df_name =="mnist" or df_name =="fashion_mnist":
        inputs = np.asfarray(data[:, 1:]) # labels are first column
        inputs = inputs/255.0 # normalize 
        labels = np.asfarray(data[:, :1])

    # transform labels into one hot representation
    n_labels = len(np.unique(labels)) 
    label_list = np.arange(n_labels)
    labels_onehot = (label_list==labels).astype(np.float64)
    
    return inputs, labels_onehot


def save_encoding(encoding, df_name, notes=None):
    relative_path = f"/config/{df_name}/"
    path = get_path("encoding", relative_path, notes)
    np.save(path, encoding)
    

def save_weights(weights, df_name, notes=None):
    relative_path = f"/config/{df_name}/weights/"
    path = get_path("config_weights", relative_path, notes)
    np.save(path, weights)
    

def load_weights(df_name, notes=None):
    relative_path = f"/config/{df_name}/weights/"
    path = get_path("config_weights", relative_path, notes)
    return np.load(path +'.npy')


def load_hyperparams(df_name):
    # opt 1: yaml
    relative_path = f"/config/{df_name}/"
    save_path = os.path.dirname(__file__) +  relative_path
    file_name = f"hyperparams.yml"
    full_path = os.path.join(save_path, file_name)
    assert os.path.exists(full_path), f'File {file_name} not found'
    with open(full_path, 'r') as config_file:
        # Converts yaml document to np array object
        params = yaml.load(config_file, Loader=UnsafeLoader)
        return params