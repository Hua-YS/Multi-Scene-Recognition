import numpy as np
from skimage.feature import hog, local_binary_pattern

def svm(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys'):
    # images should be numpy array (B x H x W x C) or (1 x H x W x C)
    if len(np.shape(images)==3):
        images = images[np.newaxis, :, :, :]


