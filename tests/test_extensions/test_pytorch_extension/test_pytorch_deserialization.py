import os
import sys

import torch
import torch.nn

from openml.extensions.pytorch import PytorchExtension
from openml.extensions.pytorch import layers
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestPytorchExtensionFlowDeserialization(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)

        self.extension = PytorchExtension()

    def test_deserialize_sequential(self):
        """ Function test_deserialize_sequential
        Test for Sequential Pytorch model deserialization
        Depends on correct implementation of model_to_flow

        :return: Nothing
        """

        sequential_orig = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[5, 5], stride=[1, 1],
                            padding=[0, 0], dilation=[1, 1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[5, 5], stride=[1, 1],
                            padding=[0, 0], dilation=[1, 1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        sequential_orig_flow = self.extension.model_to_flow(sequential_orig)

        sequential_deserialized = self.extension.flow_to_model(sequential_orig_flow)

        # we want to compare sequential_deserialized and sequential_orig
        self.assertEqual(str(sequential_deserialized), str(sequential_orig))

    def test_nested_sequential(self):
        processing_net = torch.nn.Sequential(
            layers.Functional(function=torch.Tensor.view, shape=(-1, 1, 28, 28)),
            torch.nn.BatchNorm2d(num_features=1)
        )
        features_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[5, 5], stride=[1, 1],
                            padding=[0, 0], dilation=[1, 1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[5, 5], stride=[1, 1],
                            padding=[0, 0], dilation=[1, 1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=[2, 2]),
        )
        results_net = torch.nn.Sequential(
            layers.Functional(function=torch.Tensor.view, shape=(-1, 4 * 4 * 64)),
            torch.nn.Linear(in_features=4 * 4 * 64, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=256, out_features=10),
        )
        nested_sequential_orig = torch.nn.Sequential(
            processing_net,
            features_net,
            results_net
        )

        nested_sequential_orig_flow = self.extension.model_to_flow(nested_sequential_orig)

        nested_sequential_deserialized = self.extension.flow_to_model(nested_sequential_orig_flow)

        self.assertEqual(str(nested_sequential_deserialized), str(nested_sequential_orig))
