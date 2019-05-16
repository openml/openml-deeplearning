import logging

import torch.nn
import torch.nn.functional
import torch.optim

import numpy

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
    return torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=4, gamma=1e-1)


# scheduler_gen the scheduler to be used for a given torch.optim.Optimizer
scheduler_gen = _default_scheduler_gen

# batch_size represents the processing batch size for training
batch_size = 64

# epoch_count represents the number of epochs the model should be trained for
epoch_count = 8


# _default_predict turns the outputs into probabilities using scipy.special.softmax
# and then returns the item with the highest probability
def _default_predict(output: torch.Tensor) -> numpy.ndarray:
    output_axis = output.dim() - 1
    output = output.log_softmax(dim=output_axis)
    output = torch.argmax(output, dim=output_axis)
    return output.detach().numpy()


# predict turns the outputs of the model into actual predictions
predict = _default_predict


# _default_predict_proba turns the outputs into probabilities using scipy.special.softmax
def _default_predict_proba(output: torch.Tensor) -> numpy.ndarray:
    output_axis = output.ndim - 1
    output = output.log_softmax(dim=output_axis)
    return output.detach().numpy()


# predict_proba turns the outputs of the model into probabilities for each class
predict_proba = _default_predict_proba


# _default_progress_callback reports the current epoch, step and loss for every
# training iteration to the default logger
def _default_progress_callback(epoch: int, step: int, loss: float, accuracy: float):
    logger.info('[%d, %5d] loss: %.4f, accuracy: %.4f' %
                (epoch, step, loss, accuracy))


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
    global progress_callback


_setup()
