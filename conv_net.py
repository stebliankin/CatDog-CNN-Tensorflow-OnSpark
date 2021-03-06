"""
This script is part of the FIU CAP5768 final project.

Objective:
    Convolutional Neural Network to classify cats vs dogs


"""
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils_MNIST
import time
import datetime
from PIL import Image

from image_op import get_tensor
import utils

import tensorflow.contrib.layers as lays


class ConvNet(object):
    def __init__(self, checkpoint_path, graph_path, batch_size=100, encoder=False):

#        self.desired_shape = 100
#        self.dataset_size = 15000
        self.test_percent = 0.3
        self.lr = 0.001
        self.batch_size = batch_size
        self.keep_prob = tf.constant(0.85) # used in tf.layer.dropout to leave only 85% of data to prevent overfitting
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step') # goes to the optimizer. Keeps track
                                                                        # of the batches seen so far
        self.training = True # Turn on training mode
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path

        self.num_workers=0
        self.ctx=None
        self.encoder=encoder

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_predict_data(self, tensor):
        iterator = tf.data.Iterator.from_structure(tensor.output_types,
                                                   tensor.output_shapes)

        img = iterator.get_next()
        img = tf.reshape(img, shape=[None, self.desired_shape, self.desired_shape, 1])
        self.img = tf.cast(img, tf.float32)

        self.iter = iterator.make_initializer(tensor)

    def inference(self):
        with tf.name_scope("layers"):
            if not self.encoder:

                self.conv1 = tf.layers.conv2d(inputs=self.img,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         name='conv1')
            else:
                self.autoencoder()
                self.conv1 = tf.layers.conv2d(inputs=self.last_encoder,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         padding='SAME',
                                         activation=tf.nn.relu,
                                         name='conv1')

            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool1')

            self.conv2 = tf.layers.conv2d(inputs=self.pool1,
                                     filters=64,
                                     kernel_size=[5, 5],
                                     padding='SAME',
                                     activation=tf.nn.relu,
                                     name='conv2')

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool2')

            self.conv3 = tf.layers.conv2d(inputs=self.pool2,
                                          filters=128,
                                          kernel_size=[6, 6],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          name='conv3')

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool3')

            self.conv4 = tf.layers.conv2d(inputs=self.pool3,
                                          filters=64,
                                          kernel_size=[5, 5],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          name='conv4')

            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool4')

            self.conv5 = tf.layers.conv2d(inputs=self.pool4,
                                          filters=32,
                                          kernel_size=[5, 5],
                                          padding='SAME',
                                          activation=tf.nn.relu,
                                          name='conv5')

            # pool5 = tf.layers.max_pooling2d(inputs=self.conv5,
            #                                 pool_size=[2, 2],
            #                                 strides=1,
            #                                 name='pool5')

            feature_dim = self.conv5.shape[1] * self.conv5.shape[2] * self.conv5.shape[3]

            self.conv5 = tf.reshape(self.conv5, [-1, feature_dim])
            self.fc = tf.layers.dense(self.conv5, 512, activation=tf.nn.relu, name='fc') # fully connected layer
            self.dropout = tf.layers.dropout(self.fc,
                                        self.keep_prob,
                                        training=self.training, # Perform dropout only on training mode
                                        name='dropout')  # regularization term

            self.logits = tf.layers.dense(self.dropout, self.n_classes, name='logits')


    def autoencoder(self):
        # encoder
        # 256 x 256 x 1   ->  128 x 128 x 64
        # 128 x 128 x 64  ->  64 x 64 x 32
        # 64 x 64 x 32    ->  16 x 16 x 32
        net = lays.conv2d(self.img, 128, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 64, [5, 5], stride=2, padding='SAME')
        self.last_encoder = lays.conv2d(net, 32, [5, 5], stride=4, padding='SAME')
        # decoder
        # 2 x 2 x 8    ->  8 x 8 x 16
        # 8 x 8 x 16   ->  16 x 16 x 32
        # 16 x 16 x 32  ->  32 x 32 x 1
        net = lays.conv2d_transpose(self.last_encoder, 64, [5, 5], stride=4, padding='SAME')
        net = lays.conv2d_transpose(net, 128, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
        return net

    # def encoder(self):
    #     # encoder
    #     # 32 x 32 x 1   ->  16 x 16 x 32
    #     # 16 x 16 x 32  ->  8 x 8 x 16
    #     # 8 x 8 x 16    ->  2 x 2 x 8
    #     net1 = lays.conv2d(self.img, 32, [5, 5], stride=2, padding='SAME')
    #     net2 = lays.conv2d(net1, 16, [5, 5], stride=2, padding='SAME')
    #     net3 = lays.conv2d(net2, 8, [5, 5], stride=4, padding='SAME')
    #     return net3

    def autoencoders_loss(self):
        self.ae_outputs = self.autoencoder()  # create the Autoencoder network
        self.loss_autoencoders = tf.reduce_mean(tf.square(self.ae_outputs - self.img))  # claculate the mean square error loss


    def optimaze_autoencoders(self):

        self.op_autoencoders = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_autoencoders,
                                                                global_step=self.gstep)

    def show_image(self, sess, img1, img2):
        original_image, restored_image = sess.run([img1[0], img2[0]])
        original_image = np.reshape(original_image, (self.desired_shape, self.desired_shape))
        restored_image = np.reshape(restored_image, (self.desired_shape, self.desired_shape))
        Image.fromarray(original_image).show()
        Image.fromarray(restored_image).show()


    def train_autoencoder(self, epoch):
        with tf.Session() as sess:
            print('Running session')
            sess.run(tf.global_variables_initializer())
            print('Variables initialized')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path + "/autoencoders/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint has been restored")
            step = self.gstep.eval()


            for e in range(1, epoch):
                print("Running epoch {}".format(e))
                sess.run(self.train_init)
                # self.show_image(sess, self.img,  self.ae_outputs)
                # exit()
                self.training=True

                try:

                    while True:
                       _, l = sess.run([self.op_autoencoders, self.loss_autoencoders])
                       print("Loss at step {}: {}".format(step, l))
                       utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                       'Loss at step {0}: {1}.'.format(step, l),
                                       self.log_file)
                       step+=1
                except tf.errors.OutOfRangeError:
                    print("Saving checkpoint")
                    if not os.path.exists(self.checkpoint_path):
                        os.makedirs(self.checkpoint_path)
                    saver.save(sess, self.checkpoint_path + "/autoencoders/checkpoint", step)
                    pass






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
        # optimizable_variables = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
        #                           self.pool1, self.pool2, self.pool3, self.pool4,
        #                           self.fc, self.dropout, self.logits]
        filter_vars=["conv1","conv2","conv3","conv4","conv5",
                     "pool1", "pool2","pool3","pool4", "pool5",
                     "fc", "logits", "dropout","loss"]
        train_vars=[]
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            for filter_var in filter_vars:
                if filter_var in var.name:
                    train_vars.append(var)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                            global_step=self.gstep,
                                                            var_list=train_vars)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('Training Accuracy', self.accuracy)
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
        if self.num_workers<1:
            self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()
        self.autoencoder()
        self.autoencoders_loss()
        self.optimaze_autoencoders()


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
        plt.show()


    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        # self.visualize_filters(sess)
        # exit()
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
                                    'Loss at step {0}: {1}. Training accuracy is {2}'.format(step, l,
                                                                                             sess.run(self.accuracy)),
                                self.log_file)
                step += 1
                total_loss += l
                n_batches += 1
                #self.visualize_filters(sess)
                #time.sleep(10)
        except tf.errors.OutOfRangeError:
            pass
        if self.num_workers==0:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            saver.save(sess, self.checkpoint_path + "/checkpoint", step)
            print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
            utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches),
                            self.log_file)
            print('Took: {0} minutes'.format((time.time() - start_time) / 60))
        # print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        # utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        #                 'Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches),
        #             self.log_file)
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
                if self.num_workers>1:
                    writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                num_batches+=1
        except tf.errors.OutOfRangeError:
            pass
        if epoch=="spark":
            print('Total accuracy: {} '.format(total_correct_preds / num_batches))
            utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'Total accuracy {} '.format(total_correct_preds / num_batches),
                            self.log_file)
        else:
            print('Test Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_batches))
            utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'Test Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / num_batches),
                        self.log_file)
            print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs, steps_limit=500):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        if self.num_workers==0:
            writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())

            with tf.Session() as sess:
                print('Running session')
                sess.run(tf.global_variables_initializer())
                print('Variables initialized')
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path + "/checkpoint"))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Checkpoint has been restored")
                else:
                    ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path+"/autoencoders/chekpoint"))
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print("Checkpoint from autoencoders restored")
                step = 0

                for epoch in range(n_epochs):
                    print('start training')
                   # self.visualize_filters(sess)
                    #exit()
                    step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                    self.eval_once(sess, self.test_init, writer, epoch, step)

                    #sess.run(tf.summary.image('filter', conv2))
            writer.close()
        # In case of distributed data:
        elif self.num_workers>1:
            writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            with tf.train.MonitoredTrainingSession(master=self.server.target,
                                                   is_chief=(self.task_index == 0),
                                                   scaffold=tf.train.Scaffold(init_op=init_op,
                                                                              summary_op=summary_op,
                                                                              saver=saver),
                                                   checkpoint_dir=self.checkpoint_path,
                                                   hooks=[tf.train.StopAtStepHook(last_step=steps_limit)]) as sess:
                #sess.run(tf.initializers.local_variables())
#                sess.run(tf.global_variables_initializer())
                worker_step=1
                epoch=1
                steps_at_each_epoch = int(self.max_worker_step/n_epochs)
                while not sess.should_stop():
                    if (worker_step%steps_at_each_epoch==0) and self.task_index==0:
                        self.eval_once(sess, self.test_init, writer, int(epoch), 0)
                        epoch+=1

                    if worker_step<self.max_worker_step+1:

                        sess.run(self.train_init)
                        _, l, step, accuracy = sess.run([self.opt, self.loss, self.gstep, self.accuracy])

                        utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        'Loss at step {0}: {1}. Training accuracy is {2}. Worker {3}. Task {4}'.format(step, l,
                                                                    accuracy, self.worker, self.task_index),
                                        self.log_file)

                        worker_step+=1
                    else:
                        sess.run(self.train_init)
                    # for epoch in range(n_epochs):
                    #     print('start training')
                    #     # self.visualize_filters(sess)
                    #     # exit()
                    #     step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                    #    self.eval_once(sess, self.test_init, writer, epoch, step)

    def eval_accuracy_spark(self):
        writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())
        with tf.Session() as sess:
            print('Running session')
            sess.run(tf.global_variables_initializer())
            print('Variables initialized')
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path + "/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint has been restored")
                utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'Checkpoint has been restored',
                                self.log_file)
            step = self.gstep.eval()
            self.eval_once(sess, self.test_init, writer, "spark", step)



class CatDogConvNet(ConvNet):
    def __init__(self, checkpoint_path, graph_path, dataset_size=2500, batch_size=128, log_file='log.txt',
                 num_workers=0, task_index=0, n_channels=1,
                 ctx=None, server=None, worker=None, max_worker_step=None, encoder=False):
        #super(ConvNet, self).__init__(checkpoint_path, graph_path)
        ConvNet.__init__(self, checkpoint_path, graph_path, batch_size, encoder)
        self.training_folder = "../data_catsdogs/train"
        self.desired_shape = 100
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.n_classes = 2
        self.skip_step = 40  # printing rate

        self.log_file = log_file

        self.task_index=task_index
        self.num_workers=num_workers
        self.server=server
        self.worker=worker
        self.max_worker_step=max_worker_step

        self.ctx=ctx

        self.n_channels=n_channels


    def set_dataset_size(self, size):
        self.dataset_size = size

    def get_data(self):
        with tf.name_scope('data'):
            # path, train_size, test_size, batch_size, desired_shape=300
            train_data, test_data = get_tensor(self.training_folder, int(self.dataset_size*(1-self.test_percent)),
                                               int(self.dataset_size*self.test_percent), self.batch_size,
                                               desired_shape=self.desired_shape, num_workers=self.num_workers,
                                               task_index=self.task_index, n_channels=self.n_channels)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = iterator.get_next()

            # reshape the image to make it work with tf.nn.conv2d:
            img = tf.reshape(img, shape=[-1, self.desired_shape, self.desired_shape, self.n_channels])
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
        #super(ConvNet, self).__init__(checkpoint_path, graph_path)
        ConvNet.__init__(self, checkpoint_path, graph_path)
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
    # model.train(n_epochs=6)
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
