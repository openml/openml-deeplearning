"""
MXNet sequential regression model example
==================

An example of a sequential network that solves a regression task used as an OpenML flow.
"""

import mxnet
import mxnet.gluon

import openml
import openml.extensions.mxnet

import logging

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
openml.extensions.mxnet.config.logger.setLevel(logging.DEBUG)
############################################################################

############################################################################
# Define a HybridSequential container
model = mxnet.gluon.nn.HybridSequential()
############################################################################

############################################################################
# Add the layers to the HybridSequential container
with model.name_scope():
    model.add(mxnet.gluon.nn.BatchNorm())
    model.add(mxnet.gluon.nn.Dense(units=1024, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=512, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=256, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=128, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=64, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=1, activation="relu"))
############################################################################

############################################################################
# Enable hybrid execution
model.hybridize()
############################################################################

############################################################################
# Retrieve the credit_g classification task from OpenML
task = openml.tasks.get_task(2295)
############################################################################

############################################################################
# Run the model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the run on OpenML
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
############################################################################
