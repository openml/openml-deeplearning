import mxnet

import openml
import openml.extensions.mxnet

import logging

from visdom import Visdom
import numpy as np

openml.config.logger.setLevel(logging.DEBUG)
openml.extensions.mxnet.config.logger.setLevel(logging.DEBUG)


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

    def __call__(self, fold: int, rep: int, epoch: int, step: int,
                 loss: mxnet.ndarray.NDArray,
                 metric: mxnet.metric.EvalMetric):
        loss = loss.mean().asscalar()

        for (name, value) in zip(*metric.get()):
            self.plot(name, 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                      'Class %s' % name, epoch * 984 + step, value)

        self.plot('loss', 'fold-%d-rep-%d-epoch-%d' % (fold, rep, epoch),
                  'Class loss', epoch * 984 + step, loss)


openml.extensions.mxnet.config.progress_callback = VisdomLinePlotter()

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

    task = openml.tasks.get_task(3573)
    run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
    run.publish()
