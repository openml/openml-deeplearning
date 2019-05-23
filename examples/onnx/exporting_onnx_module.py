import openml
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

# Obtain task with training data
# 3573
task = openml.tasks.get_task(10101)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

X_train[np.isnan(X_train)] = 1.0e-12
X_test[np.isnan(X_test)] = 1.0e-12

output_length = len(task.class_labels)
input_length = X_train.shape[1]

data = mx.sym.var('data')
label = mx.sym.var('softmax_label')
bnorm = mx.sym.BatchNorm(data=data)
fc1 = mx.sym.FullyConnected(data=bnorm, num_hidden=1024)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
drop1 = mx.sym.Dropout(data=act1, p=0.4)
# fc2 = mx.sym.FullyConnected(data=drop1, num_hidden=1024)
# act2 = mx.sym.Activation(data=fc2, act_type="relu")
# drop2 = mx.sym.Dropout(data=act2, p=0.4)
# fc3 = mx.sym.FullyConnected(data=drop2, num_hidden=1024)
# act3 = mx.sym.Activation(data=fc3, act_type="relu")
# drop3 = mx.sym.Dropout(data=act3, p=0.4)
# fc4 = mx.sym.FullyConnected(data=drop3, num_hidden=1024)
# act4 = mx.sym.Activation(data=fc4, act_type="relu")
# drop4 = mx.sym.Dropout(data=act4, p=0.4)
# fc5 = mx.sym.FullyConnected(data=drop4, num_hidden=1024)
# act5 = mx.sym.Activation(data=fc5, act_type="relu")
# drop5 = mx.sym.Dropout(data=act5, p=0.4)
# fc6 = mx.sym.FullyConnected(data=drop5, num_hidden=1024)
# act6 = mx.sym.Activation(data=fc6, act_type="relu")
# drop6 = mx.sym.Dropout(data=act6, p=0.4)
# fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=1024)
# act7 = mx.sym.Activation(data=fc7, act_type="relu")
# drop7 = mx.sym.Dropout(data=act7, p=0.4)
# fc8 = mx.sym.FullyConnected(data=drop7, num_hidden=1024)
# act8 = mx.sym.Activation(data=fc8, act_type="relu")
# drop8 = mx.sym.Dropout(data=act8, p=0.4)
# fc9 = mx.sym.FullyConnected(data=drop8, num_hidden=1024)
# act9 = mx.sym.Activation(data=fc9, act_type="relu")
# drop9 = mx.sym.Dropout(data=act9, p=0.4)
# fcq = mx.sym.FullyConnected(data=drop9, num_hidden=1024)
# actq = mx.sym.Activation(data=fcq, act_type="relu")
# dropq = mx.sym.Dropout(data=actq, p=0.4)
fc2 = mx.sym.FullyConnected(data=drop1, num_hidden=output_length)
mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax', label=label)
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

data_shapes = [('data', X_train.shape)]
label_shapes = [('softmax_label', y_train.shape)]

mlp_model.bind(data_shapes=data_shapes, label_shapes=label_shapes)
mlp_model.init_params(mx.init.Xavier())

mlp_model.save_params('./model-0001.params')
mlp.save('./model-symbol.json')

onnx_mxnet.export_model(
    sym='./model-symbol.json',
    params='./model-0001.params',
    input_shape=[(64, input_length)],
    onnx_file_path='model.onnx')

# model = onnx.load('model.onnx')
# print('The model is:\n{}'.format(model))
