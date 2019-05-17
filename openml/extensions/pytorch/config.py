import logging

import torch.nn
import torch.nn.functional
import torch.optim

from openml import OpenMLTask, OpenMLClassificationTask, OpenMLRegressionTask

from typing import Any, Callable

# logger is the default logger for the PyTorch extension
logger = logging.getLogger(__name__)  # type: logging.Logger

# criterion is the loss criterion used when training models
criterion = torch.nn.CrossEntropyLoss()  # type: torch.nn.Module


# _default_optimizer_gen returns the torch.optim.Adam optimizer for the given model
def _default_optimizer_gen(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(params=model.parameters())


# optimizer_gen returns the optimizer to be used for a given torch.nn.Module
optimizer_gen = _default_optimizer_gen  # type: Callable[[torch.nn.Module], torch.optim.Optimizer]


# _default_scheduler_gen returns the torch.optim.lr_scheduler.ExponentialLR
# scheduler for the given optimizer
def _default_scheduler_gen(optim: torch.optim.Optimizer) -> Any:
    return torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=1e-1)


# scheduler_gen the scheduler to be used for a given torch.optim.Optimizer
scheduler_gen = _default_scheduler_gen

# batch_size represents the processing batch size for training
batch_size = 64

# epoch_count represents the number of epochs the model should be trained for
epoch_count = 8


# _default_predict turns the outputs into probabilities using log_softmax
# and then returns the item with the highest probability
def _default_predict(output: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
    output_axis = output.dim() - 1
    if isinstance(task, OpenMLClassificationTask):
        output = output.log_softmax(dim=output_axis)
        output = torch.argmax(output, dim=output_axis)
    elif isinstance(task, OpenMLRegressionTask):
        output = output.view(-1)
    else:
        raise ValueError(task)
    return output


# predict turns the outputs of the model into actual predictions
predict = _default_predict


# _default_predict_proba turns the outputs into probabilities using log_softmax
def _default_predict_proba(output: torch.Tensor) -> torch.Tensor:
    output_axis = output.ndim - 1
    output = output.log_softmax(dim=output_axis)
    return output


# predict_proba turns the outputs of the model into probabilities for each class
predict_proba = _default_predict_proba


# _default_progress_callback reports the current epoch, step and loss for every
# training iteration to the default logger
def _default_progress_callback(epoch: int, step: int, loss: float, accuracy: float):
    logger.info('[%d, %5d] loss: %.4f, accuracy: %.4f' %
                (epoch, step, loss, accuracy))


# _default sanitizer replaces NaNs and Infs with 1e-6
def _default_sanitize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor)
    return tensor


# sanitize sanitizes the input data in order to ensure that models can be
# trained safely
sanitize = _default_sanitize


# _default_retype_labels turns the labels into torch.(cuda)LongTensor if the task is classification
# or torch.(cuda)FloatTensor if the task is regression
def _default_retype_labels(tensor: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
    if isinstance(task, OpenMLClassificationTask):
        return tensor.long()
    elif isinstance(task, OpenMLRegressionTask):
        return tensor.float()
    else:
        raise ValueError(task)


# retype_labels changes the types of the labels in order to ensure type compatibility
retype_labels = _default_retype_labels


# progress_callback is called when a training step is finished, in order to
# report the current progress
progress_callback = _default_progress_callback


def _setup():
    global logger
    global criterion
    global optimizer_gen
    global scheduler_gen
    global batch_size
    global epoch_count
    global predict
    global predict_proba
    global sanitize
    global retype_labels
    global progress_callback


_setup()
