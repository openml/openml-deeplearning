import openml
import mxnet as mx
from mxnet.contrib.onnx import onnx2mx
from mxnet import gluon, autograd
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask
from mxnet.test_utils import download

task = openml.tasks.get_task(31)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

model_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/super_resolution.onnx'
onnx_model_file = download(model_url, 'model.onnx')

# TODO: Verify this works correctly
model_mx = onnx2mx.import_to_gluon(onnx_model_file)

model_mx.initialize(mx.init.Xavier(), ctx=mx.cpu())
# Create, optimize and cache computational graph
model_mx.hybridize()

if isinstance(task, OpenMLClassificationTask):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
elif isinstance(task, OpenMLRegressionTask):
    loss_fn = gluon.loss.L2Loss()

# Initialize the trainer
# TODO: change parameters?
trainer = gluon.Trainer(model_mx.collect_params(), 'adam',
                        {'learning_rate': 0.1})

# TODO: Need to convert to NDArrays?
input = X_train
labels = y_train

# Autograd records computations done on NDArrays inside "with" block
with autograd.record():
    # Run forward propogation
    # TODO: Fix data?
    output = model_mx(input)
    loss = loss_fn(output, labels)

loss.backward()
trainer.step(input.shape[0])

pass
