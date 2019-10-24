"""
MXNet Visdom visualization example
==================

An example of how to configuration hooks can be used to draw the learning curve
and the loss in real time.
"""

import mxnet

import openml
import openml.extensions.mxnet.MXNetExtension

import logging

from visdom import Visdom
import numpy as np

openml.config.logger.setLevel(logging.DEBUG)
openml.extensions.mxnet.config.logger.setLevel(logging.DEBUG)


############################################################################
# Create a wrapper around the Vidsom communication object.
# Based on https://github.com/noagarcia/visdom-tutorial

class VisdomLinePlotter(openml.extensions.mxnet.Config):

    def __init__(self, env_name='main'):
        super().__init__()
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Iterations',
                    ylabel=var_name
                )
            )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env,
                          win=self.plots[var_name], name=split_name, update='append')

    # This is the actual interface of the progress reporting callback. The MXNet
    # extension will call this function after every training iteration with the updated
    # loss and accuracy values.
    def progress_callback(self, fold: int, rep: int, epoch: int, step: int,
                          loss: mxnet.ndarray.NDArray,
                          metric: mxnet.metric.EvalMetric):
        loss = loss.mean().asscalar()

        for (name, value) in zip(*metric.get()):
            self.plot(name, 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                      'Class %s' % name, epoch * 984 + step, value)

        self.plot('loss', 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                  'Class loss', epoch * 984 + step, loss)


############################################################################

############################################################################
# Change the default progress callback to the Visdom plotter.
openml.extensions.mxnet.config.active = VisdomLinePlotter()

############################################################################

############################################################################
# A sequential network used for classification on the MNIST dataset.
with mxnet.Context(mxnet.gpu(0)):
    model = mxnet.gluon.nn.HybridSequential()
    with model.name_scope():
        model.add(
            mxnet.gluon.nn.HybridLambda(lambda F, x: F.reshape(x, shape=(-1, 1, 28, 28))),
            mxnet.gluon.nn.BatchNorm(),
            mxnet.gluon.nn.Conv2D(channels=32, kernel_size=5),
            mxnet.gluon.nn.LeakyReLU(alpha=1e-2),
            mxnet.gluon.nn.MaxPool2D(),
            mxnet.gluon.nn.Conv2D(channels=64, kernel_size=5),
            mxnet.gluon.nn.LeakyReLU(alpha=1e-2),
            mxnet.gluon.nn.MaxPool2D(),
            mxnet.gluon.nn.Flatten(),
            mxnet.gluon.nn.Dense(units=256),
            mxnet.gluon.nn.LeakyReLU(alpha=1e-2),
            mxnet.gluon.nn.Dropout(rate=0.2),
            mxnet.gluon.nn.Dense(units=10)
        )
    ############################################################################

    ############################################################################
    # Download the OpenML task for the mnist 784 dataset.
    task = openml.tasks.get_task(3573)
    # Run the model
    run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
    run.publish()
############################################################################
