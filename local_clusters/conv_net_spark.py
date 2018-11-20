from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from datetime import datetime
from tensorflowonspark import TFCluster, TFNode

import sys

sys.path.append("..")

import conv_net

def main_fun(argv, ctx):
    # argv - parameters from sys.argv
    # ctx - node metadata like job_name, task_id
    import tensorflow as tf
    import tensorflowonspark
    import conv_net

    sys.argv = argv
    tf.app.flags.DEFINE_string('train_dir', '../../data_catsdogs/train',
                               """Directory with training images """)
    tf.app.flags.DEFINE_string('checkpoint_path', '../../checkpoints/catdog_spark',
                               """Directory with checkpoints """)
    tf.app.flags.DEFINE_string('graph_path', '../../graphs/catdog_spark',
                               """Directory with graphs """)
    tf.app.flags.DEFINE_integer('dataset_size', 25000,
                               """Dataset size """)
    tf.app.flags.DEFINE_integer('batch_size', 100,
                                """Batch Size """)
    tf.app.flags.DEFINE_integer('n_epoch', 10,
                                """Number Of Epoch """)

    FLAGS=tf.app.flags.FLAGS

    model = conv_net.CatDogConvNet(FLAGS.checkpoint_path, FLAGS.graph_path, dataset_size=FLAGS.dataset_size, batch_size=FLAGS.batch_size)
    model.training_folder = FLAGS.train_dir
    print('building a model')
    model.build()
    print('training')
    model.train(n_epochs=FLAGS.n_epoch)

if __name__ == '__main__':
  sc = SparkContext(conf=SparkConf().setAppName("catdog_spark"))
  num_executors = 2
  num_ps = 1

  cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_ps, False, TFCluster.InputMode.TENSORFLOW)
  cluster.shutdown()




