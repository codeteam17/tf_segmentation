import numpy as np
import cv2
import tensorflow as tf

# reads images from provided list of filenames
def read_dataset(input_filenames, out_filenames, n_input, size, out_size, start_idx, end_idx):

    n_out = out_size[0] * out_size[1]

    in_dataset = np.ndarray((0, n_input), dtype=np.int8)
    out_dataset = np.ndarray((0, n_out), dtype=np.int8)

    for i in range(start_idx,end_idx):

        # load input
        img = cv2.imread(input_filenames[i], 0) # load image as grayscale
        #img = cv2.resize(img, (size[0], size[1]))
        img = img.reshape([1, n_input]).astype(float)
        img /= 255.0
        #img *= 2
        #img -= 1.0
        in_dataset = np.vstack((in_dataset, img))

        # load groundtruth
        img = cv2.imread(out_filenames[i], 0) # load image as grayscale
        img = cv2.resize(img, (out_size[0], out_size[1]))
        img = img.reshape([1, n_out]).astype(float)
        img = (img / 255.0).astype(int)
        #img *= 2
        #img -= 1.0
        #img_neg = 1.0 - img
        #o = np.dstack((img, img_neg))
        out_dataset = np.vstack((out_dataset, img))
    
    return in_dataset, out_dataset

# gets the filenames from provided folders
def get_filenames(data_dir):

    input_filenames = []
    out_filenames = []

    from os import listdir
    from os.path import isfile, join

    for i in listdir(data_dir):
        file = join(data_dir, i)
        if isfile(join(data_dir, file))==False:
            continue
        if file.endswith("_out.png")==False:
            input_filenames.append(file)
        else:
            out_filenames.append(file)
    return input_filenames, out_filenames

def put_kernels_on_grid (kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 1]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 1]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8