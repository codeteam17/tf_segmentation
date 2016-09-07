
from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np

# Import MINST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 18
batch_size = 30
display_step = 1

# Network Parameters
size = (320, 240, 1)
out_size = (20, 15, 1)
n_input = size[0] * size[1] * size[2] # MNIST data input (img shape: 28*28)
n_classes = out_size[0] * out_size[1] # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


data_dir = '/Users/peric/dev/tensorflow-code/example1/data'
#data_dir = '/home/igor/dev/tf_segmentation/example1/data'
import dataset_helpers
print("Fetching file names...")
X_files, Y_files = dataset_helpers.get_filenames(data_dir)
# print("Generating datasets...")
# X, Y = dataset_helpers.read_dataset(X_files, Y_files, n_input, size, out_size)
dataset_size = len(X_files)
# print(Y.shape)


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    x = tf.nn.l2_normalize(x, 0)
    return x

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout, num_of_layers = 20):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, size[1], size[0], 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=2)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=4)

    # Convolution Layer
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=1)
    # # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)

    # # Convolution Layer
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=1)
    # # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)

    # Convolution Layer
    conv4 = tf.nn.conv2d(conv1, weights['wc4'], strides=[1, 2, 2, 1], padding='SAME')
    #conv4 = tf.nn.l2_normalize(conv4, 0)
    
    
    
    # Max Pooling (down-sampling)
    # print(prev.get_shape())
    # out = tf.reshape(prev, [-1, n_classes, 2])
    # print(out.get_shape())

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    # print(weights['wd1'].get_shape())
    # print(conv2.get_shape())
    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(out, weights['wd1']), biases['bd1'])
    #fc1 = tf.nn.softmax(out)
    # # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return conv4

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16]), name="wc1"), # out: 320x240
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32])), # 160x120
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 16])), # 40x30
    'wc4': tf.Variable(tf.random_normal([15, 20, 16, 2])), # 20x15
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([n_classes, 1])),
    # 1024 inputs, 10 outputs (class prediction)
    #'out': tf.Variable(tf.random_normal([n_classes * 2, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([16])),
    'bc4': tf.Variable(tf.random_normal([2])),
    #'bd1': tf.Variable(tf.random_normal([n_classes])),
    #'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
reshaped_logits = tf.reshape(pred, [-1, 2])  # shape [batch_size*256*256, 33]
reshaped_labels = tf.reshape(y, [-1])  # shape [batch_size*256*256]

import loss
#L = loss.loss(reshaped_logits, reshaped_labels, 2)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(reshaped_logits, reshaped_labels)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# broadcast summary for TensorBoard
# grid_x, grid_y = 4, 8   # to get a square grid for 64 conv1 features
# grid = dataset_helpers.put_kernels_on_grid (weights['wc1'], (grid_y, grid_x))
# tf.image_summary('conv1/features', grid, max_images=1)

# Evaluate model
t = tf.nn.softmax(reshaped_logits)
#t = reshaped_logits
prob_maps = tf.reshape(t, [-1, out_size[1], out_size[0], 2])
t = tf.cast(tf.argmax(prob_maps, 3), tf.int32)
out = tf.reshape(t, [-1, out_size[1], out_size[0]])
#correct_pred = tf.equal(out, tf.cast(y, tf.float32))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.Variable(1.0) # fuck it.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
global img1, img2
img1, img2 = [], []
with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph)
    global img1, img2
    sess.run(init)
    step = 0
    batch_start = 0
    batch_end = batch_size
    # Keep training until reach max iterations
    while batch_end < dataset_size:
        print("Training batch {}...".format(step))

        X, Y = dataset_helpers.read_dataset(X_files, Y_files, n_input, size, out_size, batch_start, batch_end)

        batch_x, batch_y = X, Y

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print(loss)
            print(acc)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        batch_start = batch_end
        batch_end += batch_size

        p, prob_map, logits = sess.run([ out, prob_maps, reshaped_logits ], feed_dict={x: X[:20], y: Y[:20], keep_prob: 1.0})

        print('boooooo 0')
        print(min(logits[:,0]), max(logits[:,0]))
        print('boooooo 1')
        print(min(logits[:,1]), max(logits[:,1]))

        # broadcast summary for TensorBoard
        # weights = tf.get_variable('wc1', shape=[3, 3, 1, 32])
        # tf.get_variable_scope().reuse_variables()
        # grid_x, grid_y = 4, 8   # to get a square grid for 32 conv1 features
        # grid = dataset_helpers.put_kernels_on_grid (weights, (grid_y, grid_x))
        # tf.image_summary('conv1/features', grid, max_images=1)

        img1 = p[0]
        print(img1)
        print(Y[0].reshape([out_size[1], out_size[0]]))

        print(img1.shape)
        img1 = np.float32(img1)
        img1 = cv2.resize(img1, (size[0], size[1]))

        cv2.imshow("label", img1)

        # visualize maps
        prob_0 = np.float32(prob_map[0,:,:,0])
        prob_0 = cv2.resize(prob_0, (size[0], size[1]))
        prob_1 = np.float32(prob_map[0,:,:,1])
        prob_1 = cv2.resize(prob_1, (size[0], size[1]))
        cv2.imshow("prob_0", prob_0)
        cv2.imshow("prob_1", prob_1)

        cv2.waitKey(10)

        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 20 images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X[:20], y: Y[:20], keep_prob: 1.}))
    
    