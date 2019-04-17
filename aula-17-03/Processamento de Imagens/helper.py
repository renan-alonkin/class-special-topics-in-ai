import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels = None, n_rows = 1, figsize=(25, 10) ):
    fig = plt.figure(figsize=figsize)
    n_images = len(images)
    for idx in range(n_images):
        ax = fig.add_subplot(n_rows, np.ceil(n_images/n_rows), idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        if (labels is not None):
            ax.set_title(str(labels[idx]))
            ax.title.set_fontsize(16)
            
            
def show_image_in_details(img, fig_size = (12,12)):
    fig = plt.figure(figsize=fig_size) 
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_aspect('auto')
    img = img.numpy()
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = np.round(img[x][y],2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',                        
                        color='white' if img[x][y]<thresh else 'black')
            
            
def show_images_rgb(images, labels = None, n_rows = 1, figsize=(25, 10) ):
    fig = plt.figure(figsize=figsize)
    n_images = len(images)
    for idx in range(n_images):
        ax = fig.add_subplot(n_rows, np.ceil(n_images/n_rows), idx+1, xticks=[], yticks=[])
        image_transposed = np.transpose(images[idx], (1,2,0) )
        ax.imshow( image_transposed )
        if (labels is not None):
            ax.set_title(str(labels[idx]))
            ax.title.set_fontsize(16)