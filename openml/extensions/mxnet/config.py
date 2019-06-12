import logging

import mxnet.gluon

from openml import OpenMLTask, OpenMLClassificationTask, OpenMLRegressionTask

from typing import Union, Callable, Optional

# logger is the default logger for the MXNet extension
logger = logging.getLogger(__name__)  # type: logging.Logger


# _default_criterion_gen returns a loss criterion based on the task type - regressions use
#  mxnet.gluon.loss.L1Loss while classifications use mxnet.gluon.loss.SoftmaxCrossEntropyLoss
def _default_criterion_gen(task: OpenMLTask) -> mxnet.gluon.loss.Loss:
    if isinstance(task, OpenMLRegressionTask):
        return mxnet.gluon.loss.L1Loss()
    elif isinstance(task, OpenMLClassificationTask):
        return mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    else:
        raise ValueError(task)


# loss_gen returns the loss criterion based on the task type
criterion_gen = _default_criterion_gen  # type: Callable[[OpenMLTask], mxnet.gluon.loss.Loss]


# _default_scheduler_gen returns the None scheduler for a given task
def _default_scheduler_gen(_: OpenMLTask) -> 'Optional[mxnet.lr_scheduler.LRScheduler]':
    return None


# scheduler_gen returns the scheduler to be used for a given task
scheduler_gen = _default_scheduler_gen \
    # type: Callable[[OpenMLTask], Optional[mxnet.lr_scheduler.LRScheduler]]


# _default_optimizer_gen returns the mxnet.optimizer.Adam optimizer for the given task
def _default_optimizer_gen(lr_scheduler: mxnet.lr_scheduler.LRScheduler, _: OpenMLTask) \
        -> mxnet.optimizer.Optimizer:
    return mxnet.optimizer.Adam(lr_scheduler=lr_scheduler)


# optimizer_gen returns the optimizer to be used for a given OpenMLTask
optimizer_gen = _default_optimizer_gen \
    # type: Callable[[mxnet.lr_scheduler.LRScheduler, OpenMLTask], mxnet.optimizer.Optimizer]

# batch_size represents the processing batch size for training
batch_size = 64  # type: int

# epoch_count represents the number of epochs the model should be trained for
epoch_count = 32  # type: int

# backend_type represents the possible MXNet backends
backend_type = Union[mxnet.ndarray.NDArray, mxnet.symbol.Symbol]


# _default_predict turns the outputs into predictions by returning the argmax of the output
# for classification tasks, and by flattening the prediction in case of the regression
def _default_predict(output: backend_type,
                     task: OpenMLTask) -> 'backend_type':
    if isinstance(task, OpenMLClassificationTask):
        output_axis = len(output.shape) - 1
        output = output.argmax(axis=output_axis)
        output = output.astype('int')
    elif isinstance(task, OpenMLRegressionTask):
        output = output.reshape(shape=(-1,))
    else:
        raise ValueError(task)
    return output


# predict turns the outputs of the model into actual predictions
predict = _default_predict  # type: Callable[[backend_type, OpenMLTask], backend_type]


# _default_predict_proba turns the outputs into probabilities using softmax
def _default_predict_proba(output: backend_type) -> backend_type:
    output_axis = len(output.shape) - 1
    output = output.softmax(axis=output_axis)
    return output


# predict_proba turns the outputs of the model into probabilities for each class
predict_proba = _default_predict_proba  # type: Callable[[backend_type], backend_type]


# _default sanitizer replaces NaNs with 1e-6
def _default_sanitize(output: mxnet.ndarray.NDArray) -> mxnet.ndarray.NDArray:
    output = mxnet.ndarray.where(mxnet.ndarray.contrib.isnan(output),
                                 mxnet.ndarray.ones_like(output) * 1e-6,
                                 output)
    return output


# sanitize sanitizes the input data in order to ensure that models can be
# trained safely
sanitize = _default_sanitize  # type: Callable[[mxnet.ndarray.NDArray], mxnet.ndarray.NDArray]


# _default_metric_gen returns a composite metric composed of accuracy for classification tasks,
# and composed of mean squared error, mean absolute error
# and squared mean squared error for regression tasks
def _default_metric_gen(task: OpenMLTask) -> 'mxnet.metric.EvalMetric':
    if isinstance(task, OpenMLClassificationTask):
        return mxnet.metric.CompositeEvalMetric(metrics=[mxnet.metric.Accuracy()])
    elif isinstance(task, OpenMLRegressionTask):
        return mxnet.metric.CompositeEvalMetric(metrics=[mxnet.metric.MSE(),
                                                         mxnet.metric.MAE(),
                                                         mxnet.metric.RMSE()])
    else:
        raise ValueError(task)


# metric_gen returns the metric to be used for the given task
metric_gen = _default_metric_gen  # type: Callable[[OpenMLTask], mxnet.metric.EvalMetric]


# _default_progress_callback reports the current fold, rep, epoch, step, loss
# and metric for every training iteration to the default logger
def _default_progress_callback(fold: int, rep: int, epoch: int, step: int,
                               loss: mxnet.ndarray.NDArray,
                               metric: mxnet.metric.EvalMetric):
    loss = loss.mean().asscalar()

    metric_result = ', '.join('%s: %s' % (name, value) for (name, value) in zip(*metric.get()))

    logger.info('[%d, %d, %d, %d] loss: %.4f, %s' %
                (fold, rep, epoch, step, loss, metric_result))


# progress_callback is called when a training step is finished, in order to
# report the current progress
progress_callback = _default_progress_callback \
    # type: Callable[[int, int, int, int, mxnet.ndarray.NDArray, mxnet.metric.EvalMetric], None]


def _setup():
    global logger
    global criterion_gen
    global optimizer_gen
    global scheduler_gen
    global batch_size
    global epoch_count
    global predict
    global predict_proba
    global sanitize
    global metric_gen
    global progress_callback


_setup()
