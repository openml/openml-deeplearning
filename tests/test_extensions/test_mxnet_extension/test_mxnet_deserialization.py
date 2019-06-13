import os
import sys
import json

import mxnet
from mxnet.gluon import nn
from openml.extensions.mxnet import MXNetExtension
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestMXNetExtensionFlowDeserialization(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)

        self.extension = MXNetExtension()

    def test_deserialize_sequential(self):

        # Define a basic HybridSequential model with 2 output neurons
        model = nn.HybridSequential()
        with model.name_scope():
            model.add(
                nn.Dense(1024, activation="relu"),
                nn.Dense(2)
            )

        # Enable hybrid execution
        model.hybridize()

        placeholder_input = mxnet.sym.var('data')

        symbolic_orig = model(placeholder_input)
        symbolic_orig_dict = json.loads(symbolic_orig.tojson())

        # Convert the model to an OpenMLFlow
        sequential_orig_flow = self.extension.model_to_flow(model)

        # Convert the obtained OpenMLFlow back to an MXNet model
        sequential_deserialized = self.extension.flow_to_model(sequential_orig_flow)

        # Apply the placeholder symbol input on the deserialized model
        symbolic_deserialized = sequential_deserialized(placeholder_input)
        symbolic_deserialized_dict = json.loads(symbolic_deserialized.tojson())

        # Test whether the original model and the deserialized model are identical
        self.assertDictEqual(symbolic_orig_dict, symbolic_deserialized_dict)
