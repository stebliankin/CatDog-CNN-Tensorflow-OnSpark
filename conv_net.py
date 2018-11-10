"""
This script is part of the FIU CAP5768 final project.

Objective:
    Convolutional Neural Network to classify cats vs dogs

The structure of the model and part of the code was taken from
the open sourced Stanford course "CS 20: Tensorflow for Deep Learning Research"
(https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/07_convnet_mnist.py)

"""
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import pylab
import matplotlib.pyplot as plt
import utils_MNIST
import time
import datetime

from image_op import get_tensor
import utils

class ConvNet(object):
    def __init__(self, checkpoint_path, graph_path):

        self.test_percent = 0.2
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.85) # used in tf.layer.dropout to leave only 85% of data to prevent overfitting
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step') # goes to the optimizer. Keeps track
                                                                        # of the batches seen so far
        self.training = True # Turn on training mode
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path

    def get_predict_data(self, tensor):
        iterator = tf.data.Iterator.from_structure(tensor.output_types,
                                                   tensor.output_shapes)

        img = iterator.get_next()
        img = tf.reshape(img, shape=[-1, self.desired_shape, self.desired_shape, 1])
        self.img = tf.cast(img, tf.float32)

        self.iter = iterator.make_initializer(tensor)

    def inference(self):
        self.conv1 = tf.layers.conv2d(inputs=self.img,
                                 filters=32,
                                 kernel_size=[6, 6],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=self.conv1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool1')

        self.conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[6, 6],
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=self.conv2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool2')

        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc') # fully connected layer
        dropout = tf.layers.dropout(fc,
                                    self.keep_prob,
                                    training=self.training, # Perform dropout only on training mode
                                    name='dropout')  # regularization term

        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

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
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            if self.n_classes > 1:
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            else:
                preds_round = tf.cast(tf.round(preds), tf.int64)
                correct_preds = tf.equal(preds_round, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        #self.get_data()
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def visualize_filters(self, sess):
        filters = sess.run(self.conv1)
        num_filters = filters.shape[3]
        plt.figure(1, figsize=(14,14))
        n_columns = 20
        n_rows = int(num_filters/n_columns) +1
        for i in range(num_filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter'+str(i))
            plt.imshow(filters[0,:,:,i], interpolation='nearest', cmap='gray')
        pylab.show()


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
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
                #self.visualize_filters(sess)
                time.sleep(10)
        except tf.errors.OutOfRangeError:
            pass
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        saver.save(sess, self.checkpoint_path, step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} minutes'.format((time.time() - start_time)/60))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        num_batches = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                num_batches+=1
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_batches))
        utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_batches),
                        "log.txt")
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())

        with tf.Session() as sess:
            print('Running session')
            sess.run(tf.global_variables_initializer())
            print('Variables initialized')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                print('start training')
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)

                #sess.run(tf.summary.image('filter', conv2))
        writer.close()


class CatDogConvNet(ConvNet):
    def __init__(self, checkpoint_path, graph_path):
        super().__init__(checkpoint_path, graph_path)
        self.training_folder = "../data_catsdogs/train"
        self.desired_shape = 100
        self.dataset_size = 15000

        self.n_classes = 2
        self.skip_step = 20  # printing rate



    def get_data(self):
        with tf.name_scope('data'):
            # path, train_size, test_size, batch_size, desired_shape=300
            train_data, test_data = get_tensor(self.training_folder, int(self.dataset_size*(1-self.test_percent)),
                                               int(self.dataset_size*self.test_percent), self.batch_size,
                                               desired_shape=self.desired_shape)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = iterator.get_next()

            # reshape the image to make it work with tf.nn.conv2d:
            img = tf.reshape(img, shape=[-1, self.desired_shape, self.desired_shape, 1])
            self.img = tf.cast(img, tf.float32)

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data
    def predict(self, tensor):
        # input: tensor of images to predict with shape = (number of images, img width, img height)
        # output: list of predicted classes
        self.get_predict_data(tensor)
        self.inference()

        with tf.Session() as sess:
            sess.run(self.iter)
            print('Running session')
            sess.run(tf.global_variables_initializer())
            print('Variables initialized')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("checkpoint has been restored")

            # sess.run(self.img)
            # print(sess.run(self.img))
            lbl_predicted = []
            try:
                while True:
                    prediction = sess.run(self.logits)
                    if np.argmax(prediction) == 0:
                        lbl = 1
                    else:
                        lbl = 0
                    lbl_predicted.append(lbl)
            except tf.errors.OutOfRangeError:
                pass
        return lbl_predicted

class MnistConvNet(ConvNet):
    def __init__(self, checkpoint_path, graph_path):
        super().__init__(checkpoint_path, graph_path)
        self.training_folder = "data/mnist"
        self.desired_shape = 28
        self.dataset_size = 60000
        self.n_classes = 10
        self.skip_step = 1  # printing rate

    def get_data(self):
        with tf.name_scope("mnist_data"):
            train_data, test_data = utils_MNIST.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def predict(self, tensor):
        # input: tensor of images to predict with shape = (number of images, img width, img height)
        # output: list of predicted classes
        self.get_predict_data(tensor)
        self.inference()

        with tf.Session() as sess:
            sess.run(self.iter)
            print('Running session')
            sess.run(tf.global_variables_initializer())
            print('Variables initialized')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path))
            print(self.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Restored")
            # sess.run(self.img)
            # print(sess.run(self.img))
            lbl_predicted = []
            try:
                while True:
                    prediction = sess.run(self.logits)
                    lbl = np.argmax(prediction)
                    lbl_predicted.append(lbl)
            except tf.errors.OutOfRangeError:
                pass
        return lbl_predicted

#if __name__ == '__main__':
    # start = time.time()
    # print('start program')
    # model = MnistConvNet()
    # print('building a model')
    # model.build()
    # print('training')
    # model.train(n_epochs=1)
    # print("Done. Running time is {} min.".format((time.time() - start)/60))
    # Read pixels of image:
    # import cv2
    #
    # img = cv2.imread("MNIST_testing/2.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (28, 28))
    # img = img.reshape([1, 28, 28])
    # # tf_image = tf.image.decode_png(workdir+"/MNIST_testing/9.png", channels=1)
    # # image_float = tf.image.convert_image_dtype(tf_image, tf.float32)
    # # image_float = tf.reshape(image_float, shape = [-1, 28, 28, 1])
    # # image_float = tf.cast(image_float, tf.float32)
    # tensor = tf.data.Dataset.from_tensor_slices(img)
    # # tensor = tf.reshape(tensor, shape=[1,28,28,])
    # checkpoint_path = "/Users/stebliankin/Desktop/Data Science-CAP5768/project/CatDog-CNN-Tensorflow-OnSpark/checkpoints/conv_layers-430"
    # graph_path = "/Users/stebliankin/Desktop/Data\ Science-CAP5768/project/graphs/convnet_layers"
    #
    # model = MnistConvNet(checkpoint_path=checkpoint_path, graph_path=graph_path)
    # #model.build()
    # print("Predicted number is {}".format(model.predict(tensor)))
