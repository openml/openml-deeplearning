import openml
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask

# Obtain task
task = openml.tasks.get_task(3573)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

sym, arg, aux = onnx_mxnet.import_model('model_module.onnx')
# data_names = [graph_input for graph_input in sym.list_inputs()
#               if graph_input not in arg and graph_input not in aux]
# model_mx = mx.mod.Module(symbol=sym, data_names=data_names, label_names=['softmax_label',],
#                          context=mx.cpu())
batch_size = 32
train_iter = mx.io.NDArrayIter(data={'data': nd.array(X_train)},
                               label={'softmax': nd.array(y_train)},
                               batch_size=batch_size)
model_mx = mx.mod.Module(symbol=sym,
                         data_names=['data'],
                         label_names=['softmax'],
                         context=mx.cpu())
print(train_iter.provide_label)
print(train_iter.provide_data)
model_mx.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
model_mx.set_params(arg_params=arg, aux_params=aux)

model_mx.fit(train_iter,  # train data
             optimizer='adam',  # use SGD to train
             optimizer_params={'learning_rate': 0.001},  # use fixed learning rate
             eval_metric='acc',  # report accuracy during training
             num_epoch=1)
