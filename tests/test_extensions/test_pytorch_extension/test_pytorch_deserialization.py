import os
import sys

import torch
import torch.nn

from openml.extensions.pytorch import PytorchExtension
from openml.extensions.pytorch import layers
from openml.flows.functions import assert_flows_equal
from openml.flows import OpenMLFlow
from openml.testing import TestBase

from collections import OrderedDict

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


def remove_component_id(flow: 'OpenMLFlow') -> 'OpenMLFlow':
    '''
    Removes the unique identifier for each component of a flow and their corresponding subcomponents
    :return: OpenMLFlow
    '''
    modified_flow_name = flow.name.rsplit('.', 1)[0]
    modified_flow_class_name = flow.class_name.rsplit('.', 1)[0]

    modified_flow_components = OrderedDict()

    for (subflow_id, subflow) in flow.components.items():
        subflow_name = subflow.name.rsplit('.', 1)[0]
        subflow_class_name = subflow.class_name.rsplit('.', 1)[0]
        subflow = remove_component_id(subflow)
        modified_flow_component = OpenMLFlow(name=subflow_name,
                                             class_name=subflow_class_name,
                                             description=subflow.description,
                                             model=subflow.model,
                                             components=subflow.components,
                                             parameters=subflow.parameters,
                                             parameters_meta_info=subflow.parameters_meta_info,
                                             external_version=subflow.external_version,
                                             tags=subflow.tags,
                                             language=subflow.language,
                                             dependencies=subflow.dependencies)
        modified_flow_components[subflow_id] = modified_flow_component

    modified_flow = OpenMLFlow(name=modified_flow_name,
                               class_name=modified_flow_class_name,
                               description=flow.description,
                               model=flow.model,
                               components=modified_flow_components,
                               parameters=flow.parameters,
                               parameters_meta_info=flow.parameters_meta_info,
                               external_version=flow.external_version,
                               tags=flow.tags,
                               language=flow.language,
                               dependencies=flow.dependencies
                               )

    return modified_flow


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
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        sequential_orig_flow = self.extension.model_to_flow(sequential_orig)

        sequential_deserialized = self.extension.flow_to_model(sequential_orig_flow)

        sequential_deserialized_flow = self.extension.model_to_flow(sequential_deserialized)
        sequential_deserialized_flow = remove_component_id(sequential_deserialized_flow)

        sequential_orig_flow = self.extension.model_to_flow(sequential_orig)
        sequential_orig_flow = remove_component_id(sequential_orig_flow
                                                   )

        # we want to compare sequential_deserialized and sequential_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            sequential_deserialized_flow,
            sequential_orig_flow,
        )

    def test_nested_sequential(self):
        processing_net = torch.nn.Sequential(
            layers.Reshape((-1, 1, 28, 28)),
            torch.nn.BatchNorm2d(num_features=1)
        )
        features_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        results_net = torch.nn.Sequential(
            layers.Reshape((-1, 4 * 4 * 64)),
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

        nested_sequential_deserialized_flow = \
            self.extension.model_to_flow(nested_sequential_deserialized)
        nested_sequential_deserialized_flow = \
            remove_component_id(nested_sequential_deserialized_flow)

        sequential_orig_flow = self.extension.model_to_flow(nested_sequential_orig)
        nested_sequential_orig_flow = remove_component_id(sequential_orig_flow)

        assert_flows_equal(
            nested_sequential_deserialized_flow,
            nested_sequential_orig_flow,
        )
