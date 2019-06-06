import mxnet

import openml
import openml.extensions.mxnet

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
    # Download the OpenML task for the mnist 784 dataset.
    task = openml.tasks.get_task(3573)
    ############################################################################
    # Run the model on the task (requires an API key).
    run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
    run.publish()
