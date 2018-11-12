import conv_net_new
import time
import utils
import datetime
# Initializing checkpoints and graph path:
checkpoint_path = "../checkpoints/cat_dog-2-layers"
utils.safe_mkdir("../checkpoints")

graph_path = "../graphs/cat_dog"
dataset_size = 25000
batch_size = 100
n_epoch=30

start = time.time()
print('start program')
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Starting run_catdog_conv.py", "log.txt")
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Dataset size: {}; Batch size {}".format(dataset_size, batch_size),
                "log.txt")

model = conv_net_new.ConvNet(checkpoint_path, graph_path, dataset_size=dataset_size, batch_size=batch_size)
print('building a model')
model.build()
print('training')
model.train(n_epochs=n_epoch)
print("Done. Running time is {} min.".format((time.time() - start)/60))
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "End run_catdog_conv.py trainig of {0} Epoch \
                Running time is {1} min.".format(n_epoch, (time.time() - start)/60),
                "log.txt")
