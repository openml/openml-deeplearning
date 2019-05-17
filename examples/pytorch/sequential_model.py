import torch.nn
import torch.optim

import openml
import openml.extensions.pytorch
import openml.extensions.pytorch.layers

import logging

openml.config.logger.setLevel(logging.DEBUG)
openml.extensions.pytorch.config.logger.setLevel(logging.DEBUG)

openml.extensions.pytorch.config.epoch_count = 8
openml.extensions.pytorch.config.batch_size = 512
#openml.extensions.pytorch.config.criterion = torch.nn.SmoothL1Loss()
#openml.extensions.pytorch.config.optimizer_gen = lambda x: torch.optim.RMSprop(x.parameters())

CarNet = torch.nn.Sequential(
    openml.extensions.pytorch.layers.Reshape((-1, 1, 28, 28)),
    torch.nn.BatchNorm2d(num_features=1)
)

BaNet = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
)

TuNet = torch.nn.Sequential(
    openml.extensions.pytorch.layers.Reshape((-1, 4 * 4 * 64)),
    torch.nn.Linear(in_features=4 * 4 * 64, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=10),
    torch.nn.ReLU(),
)

AmaNet = torch.nn.Sequential(
    CarNet,
    BaNet,
    TuNet
)

TrasNet = torch.nn.Sequential(
    torch.nn.BatchNorm1d(num_features=5),
    torch.nn.Linear(in_features=5, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=128),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=128, out_features=64),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=64, out_features=32),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=32, out_features=1)
)

task = openml.tasks.get_task(3573)

run = openml.runs.run_model_on_task(AmaNet, task)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
