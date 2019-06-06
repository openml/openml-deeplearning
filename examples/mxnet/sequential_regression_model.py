import mxnet
import mxnet.gluon

import openml
import openml.extensions.mxnet

model = mxnet.gluon.nn.HybridSequential()

with model.name_scope():
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

model.hybridize()

task = openml.tasks.get_task(2295)

run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
