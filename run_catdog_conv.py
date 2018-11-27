import conv_net
import time
import utils
import datetime
# Initializing checkpoints and graph path:
checkpoint_path = "../checkpoints/catdog"
utils.safe_mkdir("../checkpoints")

graph_path = "../graphs/cat_dog"
log_file = "local_run_log.txt"
dataset_size = 8
batch_size = 2
n_epoch=15

start = time.time()
print('start program')
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Starting run_catdog_conv.py", log_file)
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Dataset size: {}; Batch size {}".format(dataset_size, batch_size),
                log_file)

model = conv_net.CatDogConvNet(checkpoint_path, graph_path, dataset_size=dataset_size, batch_size=batch_size, log_file=log_file, encoder=True)
model.desired_shape=256
print('building a model')
model.build()
print('training')
model.train(n_epochs=n_epoch)
print("Done. Running time is {} min.".format((time.time() - start)/60))
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "End run_catdog_conv.py trainig of {0} Epoch \
                Running time is {1} min.".format(n_epoch, (time.time() - start)/60),
                log_file)
