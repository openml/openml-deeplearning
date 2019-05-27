import mxnet as mx
from mxnet import gluon

from openml import OpenMLTask, OpenMLClassificationTask, OpenMLRegressionTask

from typing import Callable


# _default_criterion_gen returns a criterion based on the task type - regressions use
# gluon.loss.L2Loss while classifications use gluon.loss.SoftmaxCrossEntropyLoss
def _default_criterion_gen(task: OpenMLTask) -> gluon.loss.Loss:
    if isinstance(task, OpenMLRegressionTask):
        return gluon.loss.L2Loss()
    elif isinstance(task, OpenMLClassificationTask):
        return gluon.loss.SoftmaxCrossEntropyLoss()
    else:
        raise ValueError(task)


# criterion_gen returns the criterion based on the task type
criterion_gen = _default_criterion_gen  # type: Callable[[OpenMLTask], gluon.loss.Loss]


# optimizer represents the optimizer to be used during training of the model
optimizer = mx.optimizer.Adam()  # type: mx.optimizer.Optimizer

# batch_size represents the processing batch size for training
batch_size = 32  # type: int

# epoch_count represents the number of epochs the model should be trained for
epoch_count = 32  # type: int

# sanitize represents the number that will be used to replace NaNs in train and test data
sanitize_value = 1e-6  # type: float

# context represents the context of the MXNet model - by default it will use the CPU
context = mx.cpu()


def _setup():
    global criterion_gen
    global optimizer
    global batch_size
    global epoch_count
    global sanitize_value
    global context


_setup()
