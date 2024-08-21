import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from scipy.stats import gaussian_kde


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

# plotting the results
def plot_iou_dist(thr_met, thr_otsu, thr_min):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.hist(thr_met, bins=20, alpha=0.5)  
    kde=gaussian_kde(thr_met, 0.2)
    ax.plot(np.linspace(0, 1, 100), kde(np.linspace(0, 1, 100)), color='grey', label='kde')
    ax.axvline(thr_otsu, color='C1', label='otsu', alpha=0.5)
    ax.axvline(thr_min, color='C2', label='min', alpha=0.5)
    ax.set_title('IOU dist. for matched ROIs')
    ax.legend()

def plot_all_roi_overlay(roi_image_both, roi_image_both_matched, roi_image_both_matched_thr):
    
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    ax.imshow(roi_image_both, alpha=0.5)
    ax.axis('off')
    ax.set_title('Overlay of ROIs between cp_t2p (r) and s2p (g)')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    ax.imshow(roi_image_both_matched, alpha=0.5)
    ax.set_title('Match of ROIs between cp_t2p (r) and s2p (g)')
    ax.axis('off')

    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    ax.imshow(roi_image_both_matched, alpha=0.5)
    ax.imshow(roi_image_both_matched_thr, alpha=0.5)
    ax.set_title('Thresholding of matches between cp_t2p (r) and s2p (g) \n faint: below thr. \n strong: above thr.')
    ax.axis('off')

def roi_to_img_2ch(roi_ch1, roi_ch2, rgb_ch1=0, rgb_ch2=1):
    
    roi_image = np.zeros((roi_ch1[0].shape[0], roi_ch1[0].shape[0], 4))
    # set transparency channel to fully opaque
    roi_image[:, :, 3] = 1

    # roi intensity
    roi_int = 0.8

    # now generate an image with rois in green and the background transparent
    for i in range(roi_ch1.shape[2]):
        this_roi = roi_ch1[:, :, i].astype(float)
        # now add the roi to the green channel
        roi_image[:, :, rgb_ch1] += this_roi
        # now lower the transparency from the transparency channel by a bit in a way that the overlap between rois will look darker
        roi_image[:, :, 3] -= this_roi * (1-roi_int)


    # now for the other channel
    for i in range(roi_ch2.shape[2]):
        this_roi = roi_ch2[:, :, i].astype(float)
        # now add the roi to the green channel
        roi_image[:, :, rgb_ch2] += this_roi
        # now lower the transparency from the transparency channel by a bit in a way that the overlap between rois will look darker
        roi_image[:, :, 3] -= this_roi * (1-roi_int)


    # now set all the transparent pixels to 0 and white
    roi_image[roi_image[:, :, 3] == 1, :] = 1
    roi_image[roi_image[:, :, 3] == 0, 3] = 0

    return roi_image