import os
import onnx
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

import openml
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask


ONNX_FILE_PATH_DEFAULT = 'model.onnx'
MXNET_PARAMS_PATH_DEFAULT = './model-0001.params'
MXNET_SYMBOL_PATH_DEFAULT = './model-symbol.json'


def explicit_model():
    task = openml.tasks.get_task(10101)
    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0)
    X_train = X[train_indices]
    X_test = X[test_indices]

    X_train[np.isnan(X_train)] = 1.0e-12
    X_test[np.isnan(X_test)] = 1.0e-12

    output_length = len(task.class_labels)
    input_length = X_train.shape[1]

    create_onnx_file(input_length, output_length, X_train, task)
    model = onnx.load_model(ONNX_FILE_PATH_DEFAULT)
    return model


def create_onnx_file(input_len, output_len, X_train, task):
    data = mx.sym.var('data')
    bnorm = mx.sym.BatchNorm(data=data)
    fc1 = mx.sym.FullyConnected(data=bnorm, num_hidden=1024)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    drop1 = mx.sym.Dropout(data=act1, p=0.1)
    fc2 = mx.sym.FullyConnected(data=drop1, num_hidden=1024)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    drop2 = mx.sym.Dropout(data=act2, p=0.2)
    fc2 = mx.sym.FullyConnected(data=drop2, num_hidden=output_len)

    if isinstance(task, OpenMLClassificationTask):
        mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    elif isinstance(task, OpenMLRegressionTask):
        mlp = fc2

    mlp_model = mx.mod.Module(symbol=mlp, data_names=['data'], context=mx.cpu())

    data_shapes = [('data', X_train.shape)]

    mlp_model.bind(data_shapes=data_shapes)
    init = mx.init.Xavier()
    mlp_model.init_params(initializer=init)

    mlp_model.save_params(MXNET_PARAMS_PATH_DEFAULT)
    mlp.save(MXNET_SYMBOL_PATH_DEFAULT)

    onnx_mxnet.export_model(
        sym=MXNET_SYMBOL_PATH_DEFAULT,
        params=MXNET_PARAMS_PATH_DEFAULT,
        input_shape=[(1024, input_len)],
        onnx_file_path=ONNX_FILE_PATH_DEFAULT)

    remove_mxnet_files()


def remove_mxnet_files():
    if os.path.exists(MXNET_PARAMS_PATH_DEFAULT):
        os.remove(MXNET_PARAMS_PATH_DEFAULT)

    if os.path.exists(MXNET_SYMBOL_PATH_DEFAULT):
        os.remove(MXNET_SYMBOL_PATH_DEFAULT)


def remove_onnx_file(path=ONNX_FILE_PATH_DEFAULT):
    if os.path.exists(path):
        os.remove(path)
