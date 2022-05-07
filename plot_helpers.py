import matplotlib.pyplot as plt
from collections import Counter

def show_images(images,title="",figsize=(16,9),cmap="gray"):
    """ Show images side by side in the given list. """
    image_nb = len(images)
    if image_nb == 1:
        fig,ax = plt.subplots(figsize=figsize)
        ax.imshow(images[0],cmap=cmap)
    else:
        fig,ax = plt.subplots(1,image_nb,figsize=figsize)
        for i in range(image_nb):
            ax[i].imshow(images[i],cmap=cmap)
    fig.suptitle(title,fontsize=20)

def plot_size_histogram(images,title=""):
    """ Plot the histogram of the image sizes in the given set."""    
    sizes_dict = Counter(list(map(lambda im: im.shape[:-1],images)))
    sorted_sizes = sorted(sizes_dict)
    sizes_count = [sizes_dict[s] for s in sorted_sizes] 
    fig,ax = plt.subplots(figsize=(25,9))
    ax.bar(list(map(lambda t: str(t),sorted_sizes)),sizes_count)
    plt.xticks(rotation=-60)
    ax.set_title(title,fontsize=30)