import os
import sys
from collections import OrderedDict
from unittest import mock

import torch
import torch.optim
from torch import nn

from openml import config
from openml.extensions.pytorch import PytorchExtension
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestPytorchExtensionFlowSerialization(TestBase):

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)
        self.extension = PytorchExtension()
        config.server = self.production_server

    def test_serialize_sequential_model(self):
        with mock.patch.object(self.extension, '_check_dependencies') as check_dependencies_mock:
            model = nn.Sequential(
                nn.LayerNorm(20),
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(64, 2),
                nn.Softmax(dim=0)
            )

            fixture_name = 'torch.nn.modules.container.Sequential'
            fixture_description = 'Automatically created pytorch flow.'
            version_fixture = 'torch==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                              % torch.__version__
            fixture_parameters = \
                OrderedDict([('0',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "0", "step_name": null}}'),
                             ('1',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "1", "step_name": null}}'),
                             ('2',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "2", "step_name": null}}'),
                             ('3',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "3", "step_name": null}}'),
                             ('4',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "4", "step_name": null}}'),
                             ('5',
                              '{"oml-python:serialized_object": "component_reference", '
                              '"value": {"key": "5", "step_name": null}}')])

            structure_fixture = {'torch.nn.modules.activation.ReLU': ['2'],
                                 'torch.nn.modules.activation.Softmax': ['5'],
                                 'torch.nn.modules.container.Sequential': [],
                                 'torch.nn.modules.dropout.Dropout': ['3'],
                                 'torch.nn.modules.linear.Linear': ['1'],
                                 'torch.nn.modules.linear.Linear' + str(['4']): ['4'],
                                 'torch.nn.modules.normalization.LayerNorm': ['0']}

            serialization = self.extension.model_to_flow(model)

            structure = serialization.get_structure('name')

            self.assertIn(fixture_name, serialization.name)
            self.assertEqual(serialization.class_name[:len(fixture_name)], fixture_name)
            self.assertEqual(serialization.description, fixture_description)
            self.assertEqual(serialization.parameters, fixture_parameters)
            self.assertEqual(serialization.dependencies, version_fixture)

            # Remove identifier of each component
            structure_modified = {}

            for key, value in structure.items():
                new_key = key[:key.rfind('.')]
                if new_key not in structure_modified.keys():
                    structure_modified[new_key] = value
                else:
                    structure_modified[new_key + str(value)] = value

            self.assertDictEqual(structure_fixture, structure_modified)
