from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from datetime import datetime
from tensorflowonspark import TFCluster, TFNode

import sys
import argparse


sys.path.append("..")

import conv_net

def main_fun(args, ctx):
    # ctx - node metadata like job_name, task_id

    main_path = args.main_path

    sys.path.append(main_path + "CatDog-CNN-Tensorflow-OnSpark/")

    sys.path.append(main_path+"CatDog-CNN-Tensorflow-OnSpark/")

    import tensorflow as tf
    import tensorflowonspark
    import conv_net
    import utils
    import datetime
    from image_op import get_tensor


    tf.app.flags.DEFINE_string('train_dir', main_path+'/data_catsdogs/train',
                               """Directory with training images """)
    tf.app.flags.DEFINE_string('checkpoint_path', main_path+'checkpoints/catdog_spark',
                               """Directory with checkpoints """)
    tf.app.flags.DEFINE_string('graph_path', main_path+'graphs/catdog_spark',
                               """Directory with graphs """)

    FLAGS=tf.app.flags.FLAGS

    cluster, server = TFNode.start_cluster_server(ctx)

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    log_file=main_path+"log_spark.txt"

    n_epoch = int(args.n_epoch)
    dataset_size = int(args.dataset_size)
    batch_size = int(args.batch_size)



    model = conv_net.CatDogConvNet(FLAGS.checkpoint_path, FLAGS.graph_path, dataset_size=dataset_size,
                                   batch_size=batch_size, num_workers=num_executors,
                                   task_index=task_index, ctx=ctx, server=server, worker=worker_num)
    model.training_folder = FLAGS.train_dir
    model.log_file=main_path+"log_spark.txt"
    print('building a model')
    utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Testing the model ...',
                    log_file)
    with tf.name_scope('data'):
        # path, train_size, test_size, batch_size, desired_shape=300
        train_data, test_data = get_tensor(model.training_folder,
                                           int(model.dataset_size * (1 - model.test_percent)),
                                           int(model.dataset_size * model.test_percent), model.batch_size,
                                           desired_shape=model.desired_shape, num_workers=model.num_workers,
                                           task_index=model.task_index)

       # train_data = train_data.repeat(FLAGS.n_epoch+1)

        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
        img, model.label = iterator.get_next()

        # reshape the image to make it work with tf.nn.conv2d:
        img = tf.reshape(img, shape=[-1, model.desired_shape, model.desired_shape, 1])
        model.img = tf.cast(img, tf.float32)

        model.train_init = iterator.make_initializer(train_data)  # initializer for train_data
        model.test_init = iterator.make_initializer(test_data)  # initializer for train_data
    model.build()

    print('testing')
    model.eval_accuracy_spark()



# hooks = [tf.train.StopAtStepHook(
#     last_step=int(int(self.dataset_size * (1 - self.test_percent) * n_epochs / self.batch_size)))]) as sess:

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epoch", help="number of current epoch", type=int)
    parser.add_argument("--main_path", help="Path to '../CatDog-CNN-Tensorflow-OnSpark'", required=True, type=str)
    parser.add_argument("--dataset_size", help="Training size to use", type=str)
    parser.add_argument("--batch_size", help="batch size to use", type=int)

    args = parser.parse_args()

    sc = SparkContext(conf=SparkConf().setAppName("catdog_spark"))
    num_executors = 1
    num_ps = 0


    cluster = TFCluster.run(sc, main_fun, args, num_executors, num_ps, False, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown()