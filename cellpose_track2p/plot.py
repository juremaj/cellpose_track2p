import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


def plot_fov_masks(imgs, masks, show_masks=True):
    
    nimg = len(imgs)
    fig, axs = plt.subplots(1, nimg, figsize=(10*nimg, 10), dpi=300)

    for i, (img, mask) in enumerate(zip(imgs, masks)):
        # match the histogram to the first image
        img_matched = match_histograms(img, imgs[0])
        img_matched = np.clip(img_matched, 0, np.percentile(img_matched, 74))
        img_matched = (img_matched - img_matched.min()) / (img_matched.max() - img_matched.min())

        axs[i].imshow(img_matched, cmap='gray')
        axs[i].axis('off')

        if show_masks:
            for u in np.unique(mask)[1:]:
                axs[i].contour(mask==u, [0.5], colors=[plt.cm.jet(u/len(np.unique(mask)))], linewidths=2)

def zscore(F):
    return (F - np.mean(F, axis=1)[:,np.newaxis]) / np.std(F, axis=1)[:,np.newaxis]

def plot_raster(F):
    plt.figure(figsize=(9, 3))
    plt.imshow(zscore(F), aspect='auto', cmap='Greys', vmin=0, vmax=1)
    plt.xlabel('Time (frames)')
    plt.ylabel('Neurons')
    plt.show()