from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from datetime import datetime
from tensorflowonspark import TFCluster, TFNode

import sys
import argparse


sys.path.append("..")

import conv_net

def main_fun(argv, ctx):
    # argv - parameters from sys.argv
    # ctx - node metadata like job_name, task_id

    main_path="/Users/stebliankin/Desktop/DataScience-CAP5768/project/"

    sys.path.append(main_path+"CatDog-CNN-Tensorflow-OnSpark/")

    import tensorflow as tf
    import tensorflowonspark
    import conv_net
    import utils
    import datetime

    sys.argv = argv


    tf.app.flags.DEFINE_string('train_dir', main_path+'/data_catsdogs/train',
                               """Directory with training images """)
    tf.app.flags.DEFINE_string('checkpoint_path', main_path+'checkpoints/catdog_spark',
                               """Directory with checkpoints """)
    tf.app.flags.DEFINE_string('graph_path', main_path+'graphs/catdog_spark',
                               """Directory with graphs """)
    tf.app.flags.DEFINE_integer('dataset_size', 200,
                               """Dataset size """)
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """Batch Size """)
    tf.app.flags.DEFINE_integer('n_epoch', 2,
                                """Number Of Epoch """)
    tf.app.flags.DEFINE_boolean('rdma', False, """Whether to use rdma.""")

    FLAGS=tf.app.flags.FLAGS

    cluster, server = ctx.start_cluster_server(1, FLAGS.rdma)

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    log_file=main_path+"log_spark.txt"


    if job_name == "ps":
        server.join()
        utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'PS: Merging the model',
                        log_file)

    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            model = conv_net.CatDogConvNet(FLAGS.checkpoint_path, FLAGS.graph_path, dataset_size=FLAGS.dataset_size,
                                           batch_size=FLAGS.batch_size, num_workers=num_executors,
                                           task_index=task_index, ctx=ctx, server=server)
            model.training_folder = FLAGS.train_dir
            model.log_file=main_path+"log_spark.txt"
            print('building a model')
            utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'building the model on worker {}; task index {}'.format(worker_num, task_index),
                            log_file)
            model.build()

            model.eval()
            model.summary()
            model.optimize()
            print('training')
            model.train(n_epochs=FLAGS.n_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_workers", required=True, type=str, help="Number of worker executing excluding parameter server")

    sc = SparkContext(conf=SparkConf().setAppName("catdog_spark"))
    num_executors = 3
    num_ps = 1

    cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_ps, False, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown()




