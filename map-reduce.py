import socket # for communication between machines
import tensorflow as tf

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 0))
print('listening on port:', sock.getsockname()[1])

# Define Parameter Servers and Worker Host
# tf.app.flags - the same as argparse for python
tf.app.flags.DEFINE_string("ps_hosts", "192.168.1.73:49971",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "192.168.1.73:49893, 192.168.1.73:49904, 192.168.1.73:49906",
                           "Comma-separated list of hostname:port pairs")

# Parameters of the model
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS
#Define the size of the training images
IMAGE_PIXELS=28

def weight_variable(shape):
  '''
	tf.truncated_normal returns random values from a truncated normal distribution.
	The generated values follow a normal distribution with specified mean and standard
	deviation, except that values whose magnitude is more than 2 standard deviations
	from the mean are dropped and re-picked.
	'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
These next two functions perform convolution and pooling. Convolution is the process of training on portions of 
an image, and applying the features learned from that portion to the entire image. The stride indicates 
how many pixels to shift over when applying this 'mask' to the entire image. In our case, we use the 
default of 1. Pooling is a sample based discretization process. The objective is to down-sample the input 
into bins.
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create an instance tf.train.ClusterSpec
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create server