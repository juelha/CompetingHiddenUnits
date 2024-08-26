# basics
import numpy as np
import matplotlib.pyplot as plt
import os 
import numpy as np
import yaml 
from yaml.loader import UnsafeLoader
import os

"""
Collection of 
- functions that save and load weights for the neuro fuzzy layers

"""

def load_hyperparams(df_name, best=False):
    """load weights from yaml file
    
    Args:
        layer (callable): layer the param is the attribute to
        param_name (str): name of parameter to load either: "centers", "widths", "weights
        dataset_name (str): name of datasets weights have been built on

    Raises:
        AssertionError: if save_path not found
    """
    # opt 1: yaml
   
    relative_path = f"/../config/{df_name}/"
    save_path = os.path.dirname(__file__) +  relative_path
    file_name = f"hyperparams.yml"
    full_path = os.path.join(save_path, file_name)
    assert os.path.exists(full_path), f'File {file_name} not found'
    with open(full_path, 'r') as config_file:
        # Converts yaml document to np array object
        params = yaml.load(config_file, Loader=UnsafeLoader)
        return params

    # # opt 2: np.save
    # file_name = "config_weights"
    # other_name = os.path.join(save_path, file_name)
    # loaded_weights = np.load(other_name+'.npy')
    #print("self.centers")
    #print(self.centers)
            # save params for training 


def save_weights(weights, layer_name, df_name, notes=None):
    """saves weights to yaml file
    
    Args:
        layer (callable): layer the param is the attribute to
        param_name (str): name of parameter to load either: "centers", "widths", "weights
        dataset_name (str): name of datasets weights have been built on

    Raises:
        AssertionError: if save_path not found
    """
    # save path 
    relative_path = f"/../config/{df_name}/weights/"
    save_path = os.path.dirname(__file__) + relative_path
    assert  os.path.exists(save_path), f'save_path {save_path} not found'

    # opt 1: yaml
    # file_name = f"config_{layer_name}.yaml"
    # full_path = os.path.join(save_path, file_name)
    # with open(full_path, 'w') as yaml_file:
    #     #params = getattr(layer, param_name)
    #     yaml.dump(weights.tolist(), yaml_file)

    # opt 2: np.save
    file_name = "config_weights"
    if not notes == None:
        file_name += "_" + notes
    other_name = os.path.join(save_path, file_name)
    np.save(other_name, weights)
    

def load_weights(layer_name, df_name, notes=None):
    """load weights from yaml file
    
    Args:
        layer (callable): layer the param is the attribute to
        param_name (str): name of parameter to load either: "centers", "widths", "weights
        dataset_name (str): name of datasets weights have been built on

    Raises:
        AssertionError: if save_path not found
    """
    # opt 1: yaml
    # if best:
    #     file_name = f"config_{layer_name}_best.yaml"
    # else:
    #     file_name = f"config_{layer_name}.yaml"
    # relative_path = f"/../config/{df_name}/weights/"
    # save_path = os.path.dirname(__file__) +  relative_path
    # full_path = os.path.join(save_path, file_name)
    # assert os.path.exists(full_path), f'File {file_name} not found'
    # with open(full_path, 'r') as config_file:
    #     # Converts yaml document to np array object
    #     params = yaml.load(config_file, Loader=UnsafeLoader)
    #     #setattr(layer, param_name, np.array(params))
    #     return params
  

    # # opt 2: np.save
    relative_path = f"/../config/{df_name}/weights/"
    save_path = os.path.dirname(__file__) +  relative_path
    file_name = "config_weights"
    if not notes == None:
        file_name += "_" + notes
    other_name = os.path.join(save_path, file_name)
    loaded_weights = np.load(other_name +'.npy')
  #  print("self.centers")
   # print(self.centers)
          #  save params for training 

    return loaded_weights