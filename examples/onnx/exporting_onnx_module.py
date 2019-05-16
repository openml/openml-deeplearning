import openml
import onnx
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd, sym
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask

# Obtain task with training data
task = openml.tasks.get_task(3573)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

output_length = len(task.class_labels)
input_length = X_train.shape[1]

data = mx.sym.var('data')
label = mx.sym.var('softmax_label')
fc1 = mx.sym.FullyConnected(data=data, num_hidden=64)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
drop = mx.sym.Dropout(data=act1, p=0.4)
fc2 = mx.sym.FullyConnected(data=drop, num_hidden=output_length)
mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax', label=label)
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

data_shapes = [('data', X_train.shape)]
label_shapes = [('softmax_label', y_train.shape)]

mlp_model.bind(data_shapes=data_shapes, label_shapes=label_shapes)
mlp_model.init_params()

mlp_model.save_params('./model-0001.params')
mlp.save('./model-symbol.json')

onnx_mxnet.export_model(
    sym='./model-symbol.json',
    params='./model-0001.params',
    input_shape=[(64, input_length)],
    onnx_file_path='model_module.onnx')

model = onnx.load('model_module.onnx')
print('The model is:\n{}'.format(model))
