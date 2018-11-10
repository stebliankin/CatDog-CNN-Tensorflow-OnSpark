import conv_net
import time
import utils
import datetime
# Initializing checkpoints and graph path:
checkpoints_path = "../checkpoints/catdogs_conv"
graph_path = "../graphs/catdogs_conv"

start = time.time()
print('start program')
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "Starting run_catdog_conv.py", "log.txt")
model = conv_net.MnistConvNet(checkpoints_path, graph_path)
print('building a model')
model.build()
print('training')
model.train(n_epochs=5)
print("Done. Running time is {} min.".format((time.time() - start)/60))
utils.write_log(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "End run_catdog_conv.py Running time is {} min.".format((time.time() - start)/60),
                "log.txt")
