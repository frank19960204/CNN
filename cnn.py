import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(prediction, v_xs, v_ys):
    #global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, rate: 0})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, rate: 0})
    return result

def full_connect_layer(inputs, in_size, out_size, activation_function=None, rate=0):
    # add one more layer and return the output of this layer
    Weights = weight_variable((in_size,out_size))
    biases = bias_variable((1,out_size))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
        
        #if(rate != 0 ):
        outputs = tf.nn.dropout(outputs, 1 - rate)
    else:
        outputs = activation_function(Wx_plus_b)
        #if(rate != 0 ):
        outputs = tf.nn.dropout(outputs, 1 - rate)
        
    return outputs

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1) # creat random value with normal distribution which standard deviation is 0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.compat.v1.placeholder(tf.float32, [None, 10])
rate = tf.compat.v1.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1_drop = full_connect_layer(h_pool2_flat,7*7*64,1024,tf.nn.relu,rate)


## fc2 layer ##
prediction = full_connect_layer(h_fc1_drop,1024,10,tf.nn.softmax,rate = 0)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.compat.v1.Session()

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, rate: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(prediction,
            mnist.test.images[:1000], mnist.test.labels[:1000]))
