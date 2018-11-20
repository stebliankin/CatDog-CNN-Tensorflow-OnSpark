#!/usr/bin/env bash

TFoS_HOME="/Users/stebliankin/Desktop/DataScience-CAP5768/project/TensorFlowOnSpark" #Path to TensorflowOnSpark
export TFoS_HOME=$TFoS_HOME

hostname="Vitaliis-MacBook-Pro.local"
export hostname=$hostname

# Start master:
SPARK_HOME="/Users/stebliankin/Desktop/DataScience-CAP5768/project/spark-2.4.0-bin-hadoop2.7" #directory to spark folder
export SPARK_HOME=$SPARK_HOME

export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

# Convert MNIST to RDD
cd ${TFoS_HOME}
# rm -rf examples/mnist/csv
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output examples/mnist/csv \
--format csv
ls -lR examples/mnist/csv

# Run Training
# rm -rf mnist_model
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--py-files ${TFoS_HOME}/examples/mnist/spark/mnist_dist.py \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
${TFoS_HOME}/examples/mnist/spark/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images examples/mnist/csv/train/images \
--labels examples/mnist/csv/train/labels \
--format csv \
--mode train \
--model mnist_model

ls -l mnist_model