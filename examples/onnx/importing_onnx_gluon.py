import openml
import math
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd
import numpy as np
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask, OpenMLSupervisedTask

# Obtain task
task = openml.tasks.get_task(3573)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

X_train[np.isnan(X_train)] = 1.0e-12
X_test[np.isnan(X_test)] = 1.0e-12

# Load model from onnx file, initialize it and optimize it
model_mx = onnx_mxnet.import_to_gluon('model.onnx', ctx=mx.cpu())

# Reinitialize weights and bias
model_mx.initialize(init=mx.init.Uniform(), force_reinit=True)

# Decide loss function from task type
if isinstance(task, OpenMLClassificationTask):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
elif isinstance(task, OpenMLRegressionTask):
    loss_fn = gluon.loss.L2Loss()
else:
    raise TypeError('Task not supported')

# Define trainer
trainer = gluon.Trainer(model_mx.collect_params(), 'sgd')
batch_size = 32
epochs = 20
nr_of_batches = math.ceil(X_train.shape[0] / batch_size)

for j in range(epochs):
    for i in range(nr_of_batches):
        input = nd.array(X_train[i*batch_size:(i+1)*batch_size])
        labels = nd.array(y_train[i*batch_size:(i+1)*batch_size])

        # Train the model
        with autograd.record():
            output = model_mx(input)
            loss = loss_fn(output, labels)

        loss.backward()
        trainer.step(input.shape[0])

# Predict
if isinstance(task, OpenMLSupervisedTask):
    pred_y = model_mx(nd.array(X_test))
    if isinstance(task, OpenMLClassificationTask):
        pred_y = mx.nd.argmax(pred_y, -1)
        pred_y = pred_y.asnumpy()
    if isinstance(task, OpenMLRegressionTask):
        pred_y = pred_y.asnumpy()
        pred_y = pred_y.reshape((-1))
else:
    raise ValueError(task)

print(pred_y)
