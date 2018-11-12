"""
This script is part of the FIU CAP5768 final project.

Objective:
    Convolutional Neural Network to classify cats vs dogs

The structure of the model and part of the code was taken from
the open sourced Stanford course "CS 20: Tensorflow for Deep Learning Research"
(https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/07_convnet_mnist.py)

"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' #to disable tensorflow warnings

import time
import datetime
import tensorflow as tf

# Local function:
from image_op import get_tensor
import utils

# Convolution layer with bias
def conv_relu(inputs, filters, kernel_size, stride, padding, scope_name):
    '''
    :param inputs: input images (tensor with shape [number_of_images, high, width, n_channels]) / or layer from maxpool
    :param nfilters: number of filters in convolutional layer
    :param kernel_size: size of each filter [k_size, k_size]
    :param stride: stride
    :param padding: padding
    :param scope_name: name of the module in tensorflow graph
    :return: relu activation function of convolutional layer + bias term
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel',
                                 [kernel_size, kernel_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                 [filters],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, pool_size, stride, padding="VALID", scope_name="pool"):
    '''
    :param inputs: previous layer 
    :param ksize: 
    :param stride: 
    :param padding: 
    :param scope_name: 
    :return: pooling layer
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, pool_size, pool_size, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

class ConvNet(object):
    def __init__(self, checkpoint_path, graph_path, batch_size=100, dataset_size=25000):
        self.lr = 0.001
        self.batch_size = batch_size
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.n_classes = 2
        self.skip_step = 20
        self.training = True
        self.test_percent = 0.5

        self.training_folder = "../data_catsdogs/train"
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.desired_shape = 100
        self.dataset_size = dataset_size

    def get_data(self):
        with tf.name_scope('data'):
            # path, train_size, test_size, batch_size, desired_shape=300
            train_data, test_data = get_tensor(self.training_folder, int(self.dataset_size * (1 - self.test_percent)),
                                               int(self.dataset_size * self.test_percent), self.batch_size,
                                               desired_shape=self.desired_shape)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = iterator.get_next()
            #self.label = tf.cast(self.label, tf.float32)

            # reshape the image to make it work with tf.nn.conv2d:
            img = tf.reshape(img, shape=[-1, self.desired_shape, self.desired_shape, 1])
            self.img = tf.cast(img, tf.float32)

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

    def inference(self):
        # self.conv1 = conv_relu(inputs=self.img,
        #                               filters=16,
        #                               kernel_size=4,
        #                             stride=1,
        #                               padding='SAME',
        #                               scope_name='conv1')
        #
        #
        # self.pool1 = maxpool(inputs=self.conv1,
        #                 pool_size=2,
        #                 stride=2,
        #                 scope_name='pool1')
        #
        # self.conv2 = conv_relu(inputs=self.pool1,
        #                               filters=32,
        #                               kernel_size=5,
        #                             stride=1,
        #                               padding='SAME',
        #                               scope_name='conv2')
        #
        # self.pool2 = maxpool(inputs=self.conv2,
        #                 pool_size=2,
        #                 stride=2,
        #                 scope_name='pool2')
        #
        # self.conv3 = conv_relu(inputs=self.pool2,
        #                               filters=64,
        #                               kernel_size=5,
        #                             stride=1,
        #                               padding='SAME',
        #                               scope_name='conv3')
        #
        # self.pool3 = maxpool(inputs=self.conv3,
        #                 pool_size=2,
        #                 stride=2,
        #                 scope_name='pool3')
        #
        # self.conv4 = conv_relu(inputs=self.pool3,
        #                               filters=32,
        #                               kernel_size=5,
        #                             stride=1,
        #                               padding='SAME',
        #                               scope_name='conv4')
        #
        # self.pool4 = maxpool(inputs=self.conv4,
        #                 pool_size=2,
        #                 stride=2,
        #                 scope_name='pool4')
        #
        # self.conv5 = conv_relu(inputs=self.pool4,
        #                               filters=32,
        #                               kernel_size=6,
        #                             stride=1,
        #                               padding='SAME',
        #                               scope_name='conv5')
        #
        # self.pool5 = maxpool(inputs=self.conv5,
        #                 pool_size=2,
        #                 stride=2,
        #                 scope_name='pool5')
        #
        # feature_dim = self.pool5.shape[1] * self.pool5.shape[2] * self.pool5.shape[3]
        # pool5 = tf.reshape(self.pool5, [-1, feature_dim])
        # # fc = tf.layers.dense(pool5, 1024, activation=tf.nn.relu, name='fc')  # fully connected layer
        # fc = fully_connected(pool5, 1024, 'fc')  # fully connected layer
        #
        # dropout = tf.nn.dropout(tf.nn.relu(fc),
        #                         self.keep_prob,
        #                         name='dropout')  # regularization term
        #
        # self.logits = fully_connected(dropout, self.n_classes, 'logits')

        conv1 = conv_relu(inputs=self.img,
                          filters=32,
                          kernel_size=5,
                          stride=1,
                          padding='SAME',
                          scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=64,
                          kernel_size=5,
                          stride=1,
                          padding='SAME',
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(fc), self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        compute mean cross entropy, softmax is applied internally
        '''
        #
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                            global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('Training accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            # preds_round = tf.cast(tf.round(preds), tf.int64)
            # correct_preds = tf.equal(preds_round, self.label)
           # self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.label, 1))
            correct_pred = tf.cast(correct_pred, tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)
    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}. Training accuracy is {2}'.format(step, l, sess.run(self.accuracy)))
                    utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    'Loss at step {0}: {1}. Training accuracy is {2}'.format(step, l, sess.run(self.accuracy)),
                                    "log.txt")

                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, self.checkpoint_path+"/checkpoint", step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches),
                        "log.txt")
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        summ_accuracy = 0
        n_batch = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                summ_accuracy += accuracy_batch
                n_batch += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Test accuracy at epoch {0}: {1} '.format(epoch, summ_accuracy / n_batch))
        utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Test accuracy at epoch {0}: {1} '.format(epoch, summ_accuracy / n_batch),
                        "log.txt")
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir(self.checkpoint_path)
        writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint has been restored successfully")
                utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "Checkpoint has been restored successfully",
                                "log.txt")

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

# if __name__ == '__main__':
#     checkpoint_path = "../checkpoints/cat_dog"
#     utils.safe_mkdir("../checkpoints")
#     graph_path = "../graphs/cat_dog"
#     model = ConvNet(checkpoint_path, graph_path, batch_size=100, dataset_size=25000)
#     model.build()
#     model.train(n_epochs=4)