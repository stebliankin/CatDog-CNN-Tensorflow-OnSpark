import conv_net
import utils
import time
import datetime

checkpoint_path = "../checkpoints/catdog"
graph_path = "../graphs/cat_dog"
log_file = "autoencoders_log.txt"
dataset_size = 14
batch_size = 2
n_epoch=2000

start = time.time()
print('start program')
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Starting run_catdog_conv.py", log_file)
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Dataset size: {}; Batch size {}".format(dataset_size, batch_size),
                log_file)

model = conv_net.CatDogConvNet(checkpoint_path, graph_path, dataset_size=dataset_size, batch_size=batch_size,
                               log_file=log_file, n_channels=1, encoder=True)
model.desired_shape = 256
print('building a model')
model.build()
print('training')
model.train_autoencoder(n_epoch)