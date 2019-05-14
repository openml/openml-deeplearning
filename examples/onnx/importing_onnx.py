import openml
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask

# Obtain task
task = openml.tasks.get_task(145804)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Load model from onnx file, initialize it and optimize it
model_mx = onnx_mxnet.import_to_gluon('model.onnx', ctx=mx.cpu())
# model_mx.initialize(mx.init.Xavier(), ctx=mx.cpu(), force_reinit=True)
# model_mx.hybridize()

# Decide loss function from task type
if isinstance(task, OpenMLClassificationTask):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
elif isinstance(task, OpenMLRegressionTask):
    loss_fn = gluon.loss.L2Loss()
else:
    raise TypeError('Task not supported')

# Define trainer
trainer = gluon.Trainer(model_mx.collect_params(), 'adam',
                        {'learning_rate': 0.001})

# Convert training data
input = nd.array(X_train)
labels = nd.array(y_train)

# Train the model
with autograd.record():
    # Forward propagation must be executed at least once before export
    output = model_mx(input)
    loss = loss_fn(output, labels)

loss.backward()
trainer.step(input.shape[0])

# Predict
output = model_mx(nd.array(X_test))
print(output.asnumpy())
output = mx.nd.argmax(output, -1)

pred_y = output.asnumpy()
print(pred_y)
