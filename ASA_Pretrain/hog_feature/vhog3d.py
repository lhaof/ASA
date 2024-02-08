import numpy as np
from scipy.ndimage import convolve
from skimage import draw
import matplotlib.pyplot as plt
import einops

def vhog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins, visulize, block_norm='L2'):
    """
    Inputs
    vox_volume : a 	[x x y x z] numpy array defining voxels with values in the range 0-1
    cell_size : size of a 3d cell (int)
    block_size : size of a 3d block defined in cells
    theta_histogram_bins : number of bins to break the angles in the xy plane - 180 degrees
    phi_histogram_bins : number of bins to break the angles in the xz plane - 360 degrees
    step_size : OPTIONAL integer defining the number of cells the blocks should overlap by.
	"""

    sx, sy, sz = vox_volume.shape

    num_x_cells = int(sx / cell_size)
    num_y_cells = int(sy / cell_size)
    num_z_cells = int(sz / cell_size)

    # Create 3D gradient vectors
    # X filter and vector
    x_filter = np.zeros((3, 3, 3))
    x_filter[0, 1, 1], x_filter[2, 1, 1] = 1, -1
    x_vector = convolve(vox_volume, x_filter, mode='constant', cval=0)

    # Y filter and vector
    y_filter = np.zeros((3, 3, 3))
    y_filter[1, 0, 0], y_filter[1, 2, 0] = 1, -1
    y_vector = convolve(vox_volume, y_filter, mode='constant', cval=0)

    # Z filter and vector
    z_filter = np.zeros((3, 3, 3))
    z_filter[1, 1, 0], z_filter[1, 1, 2] = 1, -1
    z_vector = convolve(vox_volume, z_filter, mode='constant', cval=0)

    magnitudes = (x_vector ** 2 + y_vector ** 2 + z_vector ** 2) ** (0.5)

    # Voxel Weights
    # kernel_size = 3
    # voxel_filter = np.full((kernel_size, kernel_size, kernel_size), 1 / (kernel_size * kernel_size * kernel_size))
    # weights = convolve(vox_volume, voxel_filter, mode='constant', cval=0)
    # weights = weights + 1

    # Gradient vector
    grad_vector = np.zeros((sx, sy, sz, 3))
    grad_vector[:, :, :, 0] = x_vector
    grad_vector[:, :, :, 1] = y_vector
    grad_vector[:, :, :, 2] = z_vector

    mag = magnitudes.copy()
    mag[np.where(magnitudes == 0)] = np.inf
    theta = np.arccos(grad_vector[:, :, :, 2] / mag)
    phi = np.abs(np.arctan2(grad_vector[:, :, :, 1], grad_vector[:, :, :, 0]))

    # Binning

    hist_pos_theta = (np.ceil(theta * theta_histogram_bins / np.pi) - 1).astype(np.int)
    hist_pos_phi = (np.ceil(phi * phi_histogram_bins / np.pi) - 1).astype(np.int)

    orientation_histogram = np.zeros((num_x_cells, num_y_cells, num_z_cells, theta_histogram_bins, phi_histogram_bins))

    for i in range(num_x_cells):
        for j in range(num_y_cells):
            for k in range(num_z_cells):
                for m in range(cell_size):
                    for n in range(cell_size):
                        for l in range(cell_size):
                            pos_t = hist_pos_theta[i * cell_size + m, j * cell_size + n, k * cell_size + l]
                            pos_p = hist_pos_phi[i * cell_size + m, j * cell_size + n, k * cell_size + l]
                            orientation_histogram[i, j, k, pos_t, pos_p] += (magnitudes[i * cell_size + m, j * cell_size + n, k * cell_size + l])# *
                                 # weights[i * cell_size + m, j * cell_size + n, k * cell_size + l])


    hog_image = None
    if visulize:
        # orientation_histogram = _hog_normalize_block(orientation_histogram, method='L2')
        radius = cell_size // 2
        orientation_arr_theta = np.arange(theta_histogram_bins)
        orientation_arr_phi = np.arange(phi_histogram_bins)

        orientation_bin_theta_midpoints = (np.pi * (orientation_arr_theta + .5) / theta_histogram_bins)
        orientation_bin_phi_midpoints = (np.pi * (orientation_arr_phi + .5) / phi_histogram_bins)

        dz_arr = radius * np.cos(orientation_bin_theta_midpoints)
        dxy_arr = radius * np.sin(orientation_bin_theta_midpoints)
        dx_arr = dxy_arr * np.sin(orientation_bin_phi_midpoints)
        dy_arr = dxy_arr * np.cos(orientation_bin_phi_midpoints)

        hog_image = np.zeros(vox_volume.shape)
        for i in range(num_x_cells):
            for j in range(num_y_cells):
                for k in range(num_z_cells):
                    for z in dz_arr:
                        for t, p, x, y in zip(orientation_arr_theta, orientation_arr_phi, dx_arr, dy_arr):
                            centre = tuple([i * cell_size + cell_size // 2,
                                            j * cell_size + cell_size // 2,
                                            k * cell_size + cell_size // 2])
                            xx1, yy1 = draw.line(int(centre[0] - x),
                                                 int(centre[1] - y),
                                                 int(centre[0] + x),
                                                 int(centre[1] + y))

                            xx2, zz2 = draw.line(int(centre[0] - x),
                                                 int(centre[2] - z),
                                                 int(centre[0] + x),
                                                 int(centre[2] + z))

                            if len(xx1) > len(xx2):
                                mul = np.ceil(len(xx1) / len(xx2))
                                zz2 = np.repeat(zz2, mul)
                                zz2 = np.random.choice(zz2, len(xx1), replace=False)
                                zz2 = np.sort(zz2)
                            elif len(xx1) < len(xx2):
                                mul = np.ceil(len(xx2) / len(xx1))
                                xx1 = np.repeat(xx1, mul)
                                yy1 = np.repeat(yy1, mul)
                                xx1 = np.random.choice(xx1, len(zz2), replace=False)
                                xx1 = np.sort(xx1)
                                yy1 = np.random.choice(yy1, len(zz2), replace=False)
                                yy1 = np.sort(yy1)

                            hog_image[xx1, yy1, zz2] += orientation_histogram[i, j, k, t, p]

                            yy3, zz3 = draw.line(int(centre[1] - y),
                                                 int(centre[2] - z),
                                                 int(centre[1] + y),
                                                 int(centre[2] + z))
                            if len(zz2) > len(zz3):
                                mul = np.ceil(len(zz2) / len(zz3))
                                zz3 = np.repeat(zz3, mul)
                                zz3 = np.random.choice(zz3, len(xx1), replace=False)
                                zz3 = np.sort(zz3)
                            elif len(zz2) < len(zz3):
                                mul = np.ceil(len(zz3) / len(zz2))
                                xx1 = np.repeat(xx1, mul)
                                yy1 = np.repeat(yy1, mul)
                                xx1 = np.random.choice(xx1, len(zz3), replace=False)
                                xx1 = np.sort(xx1)
                                yy1 = np.random.choice(yy1, len(zz3), replace=False)
                                yy1 = np.sort(yy1)

                            hog_image[xx1, yy1, zz3] += orientation_histogram[i, j, k, t, p]



    # normalize
    n_blocks_x = int(num_x_cells / block_size)
    n_blocks_y = int(num_y_cells / block_size)
    n_blocks_z = int(num_z_cells / block_size)
    normalized_blocks = np.zeros((n_blocks_x, n_blocks_y, n_blocks_z, block_size, block_size, block_size,
                                  theta_histogram_bins, phi_histogram_bins))

    for i in range(n_blocks_x):
        for j in range(n_blocks_y):
            for k in range(n_blocks_z):
                block = orientation_histogram[i * block_size:i * block_size + block_size,
                        j * block_size:j * block_size + block_size, k * block_size:k * block_size + block_size, :]
                normalized_blocks[i, j, k, :] = _hog_normalize_block(block, method=block_norm)

    if visulize:
        return normalized_blocks.ravel(), hog_image
    else:
        normalized_blocks = np.squeeze(normalized_blocks)
        normalized_blocks = einops.rearrange(normalized_blocks, "h w s t p -> (h w s)(t p)")
        return normalized_blocks


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
