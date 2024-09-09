import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, clear_output
#from Manager import get_path
"""
Collection of functions to visualize matrices
"""

def draw_weights(weights, n_cols, n_rows, df_name, fig=None, epoch=0, n_hidden=0, show=False, save=False):
    """_summary_

    Args:
        weights (np.ndarray): weight matrix of shape (n_samples, n_features)
        n_cols (_type_): _description_
        n_rows (_type_): _description_
        fig (_type_, optional): _description_. Defaults to None.
        epoch (int, optional): _description_. Defaults to 0.
        save (bool, optional): _description_. Defaults to False.
        df_name (str, optional): _description_. Defaults to None.
    """
    if fig is None:
        fig=plt.figure(figsize=(12.9,10))
    yy=0
    # harcoded for now
    if df_name =="xor":
        pxl_x = 2
        pxl_y = 1
    if df_name == "mnist" or df_name == "fashion_mnist":
        pxl_x = 28
        pxl_y = 28

    HM=np.zeros((pxl_y*n_rows,pxl_x*n_cols))
    for y in range(n_rows):
        for x in range(n_cols):
            HM[y*pxl_y:(y+1)*pxl_y, x*pxl_x:(x+1)*pxl_x]=weights[yy,:].reshape(pxl_y,pxl_x)
            yy += 1

    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.title(f"Weights at epoch: {epoch}")
    if show:
        fig.canvas.draw()   
        display(fig)
        clear_output(wait=True)
    if save:
        # get save path 
       
        file_name =  f'Weights_Epoch{epoch}_{n_hidden}hidden'  + '.png'
       # file_name =  f'fashion_mnist'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()
    fig.clear()
    plt.close(fig)
    return HM


def draw_encoding(mat, n_cols, n_rows, df_name, show=False, save=False):
    """
    n_cols (int): how many cols to display -> x
    n_rows (int): how many rows to display -> y
    """

    fig=plt.figure(figsize=(12.9,10))
    pxl_x = 10
    pxl_y = 10
    m = mat[0:pxl_y, 0:pxl_x] # take first 10x10
    plt.clf()
    nc=np.amax(np.absolute(m))
    im=plt.imshow(m,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(m), 0, np.amax(m)])
    plt.axis('off')
    plt.title(f"Encoding")
    if show:
        fig.canvas.draw()   
        display(fig) 

    if save:

        # get save path 
        file_name =  'Encoding'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()
        # display(fig)
        # clear_output(wait=True)
    fig.clear()
    plt.close(fig)
    return m


def frankensteining(weights, inputs, encoding, df_name, show=False, save=False):
    


    # Start with a square Figure.
    fig = plt.figure(figsize=(6.25, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(nrows=2, ncols=3,  
                          width_ratios=(1, 4, 0.2), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    # creating axis objs
    l = fig.add_subplot(gs[1, 0]) # left
    l.set_xticks([])
    l.set_yticks([])
    l.set_ylabel('10 Inputs')
    num = draw_weights(inputs, n_cols=1, n_rows=10, fig=plt.figure(figsize=(12.9,10)), df_name=df_name)
    max_ref = np.amax(np.absolute(num))
    im = l.imshow(num, cmap='bwr',vmin=-max_ref,vmax=max_ref)
    

    top_r = fig.add_subplot(gs[0, 1]) # top right
    top_r.set_xticks([])
    top_r.set_yticks([])
    top_r.xaxis.set_label_position('top')
    top_r.set_xlabel("Weights of 10 neurons")
    w = draw_weights(weights, n_cols=10, n_rows=1, df_name=df_name, show=False)
    top_r.imshow(w, cmap='bwr',vmin=-max_ref,vmax=max_ref)
    

    bot_r = fig.add_subplot(gs[1, 1]) # bot right
    bot_r.set_xticks([])
    bot_r.set_yticks([])
    bot_r.set_xlabel("Encoding")
    e = draw_encoding(encoding, n_cols=10, n_rows=10,df_name=df_name, show=False)
    im = bot_r.imshow(e, cmap='bwr',vmin=-max_ref,vmax=max_ref)

    bot_r_r = fig.add_subplot(gs[1, 2]) # bot right
    fig.colorbar(im, cax=bot_r_r,ticks=[np.amin(num), 0, np.amax(num)])
      
    if show:
        clear_output()
        fig.canvas.draw()   
        display(fig) 
    if save:
        # get save path 
        file_name =  'Monitoring'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
        completeName = os.path.join(save_path, file_name)

        fig.savefig(completeName)

    #.show()
    #plt.clf()
