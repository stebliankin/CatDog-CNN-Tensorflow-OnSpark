#!/usr/bin/env bash

hostname="Vitaliis-MacBook-Pro.local"

# Start master:
SPARK_HOME="/Users/stebliankin/Desktop/DataScience-CAP5768/project/spark-2.4.0-bin-hadoop2.7" #directory to spark folder
export SPARK_HOME=$SPARK_HOME

#$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/stop-master.sh
MASTER="spark://"$hostname":7077"
export MASTER=$MASTER

# You can see spark master at http://localhost:8080/

# Let's create 3 local worker machines
export SPARK_WORKER_INSTANCES=3 #export worker instances to overlap with config file

$SPARK_HOME/sbin/stop-slave.sh -c 2 -m 1G $MASTER
#$SPARK_HOME/sbin/start-slave.sh -c 2 -m 1G $MASTER

# Create one more local worker for parameter server with less cpu
#export SPARK_WORKER_INSTANCES=4
#$SPARK_HOME/sbin/start-slave.sh -c 1 -m 1G $MASTER

