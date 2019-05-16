import torch.nn

import openml
import openml.extensions.pytorch
import openml.extensions.pytorch.layers

import logging

openml.config.logger.setLevel(logging.DEBUG)
openml.extensions.pytorch.logger.setLevel(logging.DEBUG)

panarama_model = torch.nn.Sequential(
    openml.extensions.pytorch.layers.Reshape((-1, 3 * 3 * 64)),
    torch.nn.Linear(in_features=3 * 3 * 64, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=10),
    torch.nn.ReLU(),
)

main_model = torch.nn.Sequential(
    torch.nn.BatchNorm1d(num_features=1 * 28 * 28),
    openml.extensions.pytorch.layers.Reshape((-1, 1, 28, 28)),
    torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    panarama_model
)

task = openml.tasks.get_task(3573)

run = openml.runs.run_model_on_task(main_model, task)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
