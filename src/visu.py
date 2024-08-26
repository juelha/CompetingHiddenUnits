import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import fast
from IPython.display import display, clear_output


# helper function 
def draw_weights(weights, n_cols, n_rows, fig, epoch=0, save=False):
    yy=0
    HM=np.zeros((28*n_rows,28*n_cols))
    #HM=np.zeros((10*Ky,10*Kx))
    for y in range(n_rows):
        for x in range(n_cols):
            HM[y*28:(y+1)*28, x*28:(x+1)*28]=weights[yy,:].reshape(28,28)
           # HM[y*10:(y+1)*10,x*10:(x+1)*10]=synapses[yy,:].reshape(10,10)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.title(f"Weights at epoch: {epoch+1}")
    fig.canvas.draw()   
   # display(fig)
    #clear_output(wait=True)
    if save:
        # get save path 
        file_name =  f'Weights_Epoch{epoch+1}'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/figures/' 
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()

def draw_encoding(mat, n_cols, n_rows, fig, epoch=0):
    """
    n_cols (int): how many cols to display -> x
    n_rows (int): how many rows to display -> y
    """
    yy=0
    pxl_x = 10
    pxl_y = 10
    print(mat.shape)
   # HM=np.zeros((28*Ky,28*Kx))
    HM=np.zeros((n_rows*pxl_x, n_cols*pxl_y))
    for y in range(n_rows):
        for x in range(n_cols):
            #HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            HM[y*pxl_y:(y+1)*pxl_y, x*pxl_x:(x+1)*pxl_x]=mat[:,yy].reshape(pxl_y,pxl_x)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.title(f"Encoding")
    fig.canvas.draw()   

    # get save path 
    file_name =  'Encoding'  + '.png'
    save_path = os.path.dirname(__file__) +  f'/../reports/figures/' 
    completeName = os.path.join(save_path, file_name)

    plt.savefig(completeName)
    plt.clf()
    # display(fig)
    # clear_output(wait=True)