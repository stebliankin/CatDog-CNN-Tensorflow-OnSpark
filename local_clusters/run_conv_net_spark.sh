#!/usr/bin/env bash

hostname="Vitaliis-MacBook-Pro.local"

# Start master:
SPARK_HOME="/Users/stebliankin/Desktop/DataScience-CAP5768/project/spark-2.4.0-bin-hadoop2.7" #directory to spark folder
export SPARK_HOME=$SPARK_HOME

$SPARK_HOME/sbin/stop-slave.sh
$SPARK_HOME/sbin/stop-master.sh
MASTER="spark://"$hostname":7077"
export MASTER=$MASTER


$SPARK_HOME/sbin/start-master.sh
#$SPARK_HOME/sbin/stop-master.sh


# You can see spark master at http://localhost:8080/

# Let's create 3 local worker machines
export SPARK_WORKER_INSTANCES=8 #export worker instances to overlap with config file
num_ex=$(($SPARK_WORKER_INSTANCES-1))

#$SPARK_HOME/sbin/stop-slave.sh -c 2 -m 1G $MASTER
$SPARK_HOME/sbin/start-slave.sh -c 1 -m 1G $MASTER


dataset_size=25000
batch_size=100

# Train 10 epoch:

    ${SPARK_HOME}/bin/spark-submit \
    --master spark://${hostname}:7077 \
    --conf spark.cores.max=8 \
    --conf spark.task.cpus=1 \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    /Users/stebliankin/Desktop/DataScience-CAP5768/project/CatDog-CNN-Tensorflow-OnSpark/local_clusters/conv_net_spark.py \
    --num_executors $num_ex --n_epoch 10 --main_path /Users/stebliankin/Desktop/DataScience-CAP5768/project/   \
    --dataset_size $dataset_size --batch_size $batch_size

   echo Evaluating the model

    ${SPARK_HOME}/bin/spark-submit \
    --master spark://${hostname}:7077 \
    --conf spark.cores.max=4 \
    --conf spark.task.cpus=1 \
    --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
    /Users/stebliankin/Desktop/DataScience-CAP5768/project/CatDog-CNN-Tensorflow-OnSpark/local_clusters/conv_net_spark_eval.py \
    --n_epoch 10 --main_path /Users/stebliankin/Desktop/DataScience-CAP5768/project/   \
    --dataset_size $dataset_size --batch_size $batch_size








$SPARK_HOME/sbin/stop-slave.sh
$SPARK_HOME/sbin/stop-master.sh
