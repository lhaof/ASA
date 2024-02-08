import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.feature import hog
from skimage import data, color, exposure
from vhog3d import vhog3d
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import threading
from multiprocessing import Pool, cpu_count
import time
import random


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out


def vhog3d_cal(arg):
    for sub in arg:
        time_now = time.time()
        # if sub in sub_list_exist:
        #     print("Jumping ", sub)
        #     continue
        image = np.load(join(DATA_FILE, sub))["data"]
        image = normalization(image[0])
        grad_vec = vhog3d(image, cell_size, block_size, theta_histogram_bins, phi_histogram_bins, visulize=False)
        np.savez(join(HOG_SAVE, sub), data=grad_vec)

        print("Handling ", sub, "  Time:", time.time() - time_now)
    # print(arg)
    # for sub in arg[0]:
    #     print(sub)


if __name__ == "__main__":
    print(cpu_count())
    # p = Pool(60)
    # length = int(len(sub_list) / 60)
    # for i in range(60):
    #     if i == 59:
    #         sub_thr = sub_list[i * length:]
    #     else:
    #         sub_thr = sub_list[i * length:i * length + length]
    #     t = p.apply_async(vhog3d_cal, args=(sub_thr,))
    # p.close()
    # p.join()

    image = np.load(
        "Data Path/DATA_AFTERTRANS/sub-ADNI002S0619_ses-M06.npz")[
        "data"]
    image = normalization(image)
    for k in [20, 40, 50, 60, 70, 80, 100]:
        img = image[0, k, ::]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
        #
        ax1.axis('off')
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Input image')
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(16, 16), visualize=True)
        #
        region = 128
        for i in range(16):
            for j in range(16):
                hog_image[i * region:i * region + region, j * region:j * region + region] = _hog_normalize_block(
                    hog_image[i * region:i * region + region, j * region:j * region + region], method="L2")
        #
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        # #
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

        plt.imshow(hog_image_rescaled, cmap='gray')
        # plt.title('Histogram of Oriented Gradients')
        plt.axis('off')
        plt.savefig('Save Path/visual_miccai/hog_{}.jpg'.format(k), dpi=400)
        plt.show()