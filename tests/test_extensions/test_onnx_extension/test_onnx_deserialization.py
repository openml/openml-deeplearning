import os
import sys
import onnx
import openml
from .onnx_model_utils import create_onnx_file, remove_onnx_file
from openml.extensions.onnx import OnnxExtension
from openml.flows.functions import assert_flows_equal
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestOnnxExtensionFlowDeserialization(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        self.extension = OnnxExtension()

    def test_deserialize_sequential(self):
        """ Function test_deserialize_sequential_with_defaults
        Test for Sequential Keras model deserialization
        Depends on correct implementation of model_to_flow

        :return: Nothing
        """

        task_lst = [4823, 52948, 2285, 4729, 4990]

        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                # Obtain train and test data
                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices]

                # Calculate input and output shapes
                input_length = X_train.shape[1]
                output_length = 1

                # Create an ONNX file from an MXNet model
                create_onnx_file(input_length, output_length, X_train, task)

                # Load the ONNX model from the file and remove the file
                model_original = onnx.load('model.onnx')
                remove_onnx_file()

                # Create an exact copy of the model
                onnx.save(model_original, 'model_copy.onnx')
                model_adjusted = onnx.load('model_copy.onnx')
                remove_onnx_file('model_copy.onnx')

                # we want to confirm that model_original and model_adjusted are the same.
                # We use the flow equals function for this
                assert_flows_equal(
                    self.extension.model_to_flow(model_original),
                    self.extension.model_to_flow(model_adjusted),
                )

                flow = self.extension.model_to_flow(model_adjusted)
                model_deserialized = \
                    self.extension.flow_to_model(flow, initialize_with_defaults=True)

                # we want to compare model_deserialized and model_original. We use the flow
                # equals function for this
                assert_flows_equal(
                    self.extension.model_to_flow(model_original),
                    self.extension.model_to_flow(model_deserialized),
                )
