"""
MXNet sequential classification model example
==================

An example of a sequential network that does a binary classification on the credit-g dataset.
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
    model.add(mxnet.gluon.nn.Dense(units=2))
############################################################################

############################################################################
# Enable hybrid execution
model.hybridize()
############################################################################

############################################################################
# Retrieve the credit_g classification task from OpenML
task = openml.tasks.get_task(31)
############################################################################

############################################################################
# Run the model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the Run on OpenML
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
############################################################################
