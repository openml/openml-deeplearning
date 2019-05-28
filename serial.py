import mxnet
import mxnet.gluon

import openml
import openml.extensions.mxnet

import json

model = mxnet.gluon.nn.HybridSequential()

with model.name_scope():
    model.add(mxnet.gluon.nn.BatchNorm())
    model.add(mxnet.gluon.nn.Dense(units=1024, activation="relu"))
    model.add(mxnet.gluon.nn.Dropout(rate=0.4))
    model.add(mxnet.gluon.nn.Dense(units=2))

model.hybridize()

task = openml.tasks.get_task(31)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
