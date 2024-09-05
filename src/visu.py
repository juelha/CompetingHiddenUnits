import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import fast
from IPython.display import display, clear_output


# helper function 
def draw_weights(weights, n_cols, n_rows, fig, epoch=0, save=False, df_name=None):
    yy=0
    pxl_x = 28
    pxl_y = 28
    HM=np.zeros((28*n_rows,28*n_cols))
    #HM=np.zeros((10*Ky,10*Kx))
    for y in range(n_rows):
        for x in range(n_cols):
            HM[y*pxl_y:(y+1)*pxl_y, x*pxl_x:(x+1)*pxl_x]=weights[yy,:].reshape(pxl_y,pxl_x)
           # HM[y*10:(y+1)*10,x*10:(x+1)*10]=synapses[yy,:].reshape(10,10)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
   #
    plt.title(f"Weights at epoch: {epoch+1}")
   # plt.title(f"10x10 samples of FASHION MNIST")
    fig.canvas.draw()   
   # display(fig)
    #clear_output(wait=True)
    if save:
        # get save path 
       
        file_name =  f'Weights_Epoch{epoch+1}'  + '.png'
       # file_name =  f'fashion_mnist'  + '.png'
        save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()
    return HM

def draw_encoding(mat, n_cols, n_rows, fig, df_name):
    """
    n_cols (int): how many cols to display -> x
    n_rows (int): how many rows to display -> y
    """
    yy=0
    pxl_x = 10
    pxl_y = 10

    m = mat[0:pxl_y, 0:pxl_x] # take first 10x10
    sz = int(np.sqrt(mat[0,:].size))
    print("m",m[0])
    print("sss",sz)
   # HM=np.zeros((28*Ky,28*Kx))
    HM=np.zeros((n_rows,n_cols))
    for y in range(n_rows):
        for x in range(n_cols):
            print("MA",m[yy])
           # temp =m[y].reshape(-1, 1, 1)
         #   HM[y*pxl_y:(y+1)*pxl_y, y*pxl_y:(y+1)*pxl_y] = np.tile(temp,(1, 10, 10)).reshape((10,100))#.reshape( temp.shape[-1],-1)#.reshape((10,100))
            HM[y*pxl_y:(y+1)*pxl_y, :] = m[yy]
          
           # HM[y*pxl_y:(y+1)*pxl_y, x*pxl_x:(x+1)*pxl_x]=weights[yy,:].reshape(pxl_y,pxl_x)
           # HM[y*10:(y+1)*10,x*10:(x+1)*10]=synapses[yy,:].reshape(10,10)
        yy += 1


    print("before,", HM)
    #HM = HM.reshape((100,100))
    print("after,", HM)
    HM = m
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.title(f"Encoding")
    fig.canvas.draw()   

    # get save path 
    file_name =  'Encoding'  + '.png'
    save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
    completeName = os.path.join(save_path, file_name)

    plt.savefig(completeName)
    plt.clf()
    # display(fig)
    # clear_output(wait=True)
    return HM



def frankensteining(weights, inputs, encoding, df_name):
    


    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal Axes and the main Axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(nrows=2, ncols=2,  width_ratios=(1, 4), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    # Create the Axes.
    bot_r = fig.add_subplot(gs[1, 1]) # bot right
    bot_r.set_xticks([])
    bot_r.set_yticks([])
    bot_r.set_xlabel("Encoding")
  #  
    e = draw_encoding(encoding, n_cols=10, n_rows=10, fig=plt.figure(figsize=(12.9,10)), df_name=df_name)
  #  print("en", encoding)
    nc=np.amax(np.absolute(e))
    #Z = np.random.poisson(lam=6, size=(64,64))
    #bot_r.imshow(Z)

    im = bot_r.imshow(e,cmap='bwr',vmin=-nc,vmax=nc)
    #fig.colorbar(im, ax=bot_r)
    l = fig.add_subplot(gs[1, 0]) # left
    l.set_xticks([])
    l.set_yticks([])
    l.set_ylabel('10 Images')

    num = draw_weights(inputs, n_cols=1, n_rows=10, fig=plt.figure(figsize=(12.9,10)), df_name=df_name)
    l.imshow(num)

    top_r = fig.add_subplot(gs[0, 1]) # top right
    top_r.set_xticks([])
    top_r.set_yticks([])

    w = draw_weights(weights, n_cols=10, n_rows=1, fig=plt.figure(figsize=(12.9,10)), df_name=df_name)
    top_r.imshow(w)
    top_r.xaxis.set_label_position('top')
    top_r.set_xlabel("Weights of 10 neurons")
   # Draw the scatter plot and marginals.

      
    #plt.title(f"Encoding")

   # fig.canvas.draw() 

    # get save path 
    file_name =  'Monitoring'  + '.png'
    save_path = os.path.dirname(__file__) +  f'/../reports/{df_name}/figures/' 
    completeName = os.path.join(save_path, file_name)

    fig.savefig(completeName)
    print(encoding.shape)

    #.show()
    #plt.clf()
