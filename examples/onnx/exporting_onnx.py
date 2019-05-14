import openml
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet import nd, gluon, autograd
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask

# Obtain task with training data
task = openml.tasks.get_task(145804)
X, y = task.get_X_and_y()
train_indices, test_indices = task.get_train_test_split_indices(
    repeat=0, fold=0, sample=0)
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

output_length = len(task.class_labels)
input_length = X_train.shape[1]

# Create simple sequential model
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1024, activation="relu"))
    net.add(gluon.nn.Dropout(0.4))
    net.add(gluon.nn.Dense(output_length, activation="softrelu"))

# Initialize and optimize the model
# TODO: Initializer?
net.initialize(mx.init.Xavier(), ctx=mx.cpu())
net.hybridize()

# Check task to determine loss function
if isinstance(task, OpenMLClassificationTask):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
elif isinstance(task, OpenMLRegressionTask):
    loss_fn = gluon.loss.L2Loss()
else:
    raise TypeError('Task not supported')

# Define trainer
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': 0.1})

# Convert training data
input = nd.array(X_train)
labels = nd.array(y_train)

# Train the model
with autograd.record():
    # Forward propagation must be executed at least once before export
    output = net(input)
#     loss = loss_fn(output, labels)
#
# loss.backward()
# trainer.step(input.shape[0])

# Export model
net.export('model', epoch=1)
onnx_mxnet.export_model(
    'model-symbol.json',
    'model-0001.params',
    input_shape=[(1024, input_length)],
    onnx_file_path='model.onnx')
