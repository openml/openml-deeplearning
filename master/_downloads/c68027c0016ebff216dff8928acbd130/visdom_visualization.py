"""
PyTorch Visdom visualization example
==================

An example of how to configuration hooks can be used to draw the learning curve
and the loss in real time.
"""

import torch.nn

import openml
import openml.extensions.pytorch
import openml.extensions.pytorch.layers

from visdom import Visdom

import numpy as np
import math


############################################################################
# Create a wrapper around the Vidsom communication object.
# Based on https://github.com/noagarcia/visdom-tutorial
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]),
                                                 env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env,
                          win=self.plots[var_name], name=split_name, update='append')

    # This is the actual interface of the progress reporting callback. The PyTorch
    # extension will call this function after every training iteration with the updated
    # loss and accuracy values.
    def __call__(self, fold: int, rep: int, epoch: int, step: int, loss: float, accuracy: float):
        self.plot('loss', 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                  'Class Loss', epoch * 984 + step, loss)

        # Plot the accuracy only if it is present.
        if not math.isnan(accuracy):
            self.plot('accuracy', 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                      'Class Accuracy', epoch * 984 + step, accuracy)
############################################################################


############################################################################
# Change the default progress callback to the Visdom plotter.
openml.extensions.pytorch.config.progress_callback = VisdomLinePlotter()
############################################################################

############################################################################
# The network described in the sequential classification example.
model = torch.nn.Sequential(
    openml.extensions.pytorch.layers.Reshape((-1, 1, 28, 28)),
    torch.nn.BatchNorm2d(num_features=1),
    torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    torch.nn.LeakyReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    torch.nn.LeakyReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    openml.extensions.pytorch.layers.Reshape((-1, 4 * 4 * 64)),
    torch.nn.Linear(in_features=4 * 4 * 64, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=10),
)
############################################################################

############################################################################
# Download the OpenML task for the mnist 784 dataset.
task = openml.tasks.get_task(3573)
############################################################################
# Run the model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

############################################################################
