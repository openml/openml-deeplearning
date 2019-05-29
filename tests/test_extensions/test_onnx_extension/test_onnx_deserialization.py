import os
import sys
import inspect
import onnx

import openml
from openml.testing import TestBase
from openml.extensions.onnx import OnnxExtension
from openml.flows.functions import assert_flows_equal

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestOnnxExtensionFlowDeserialization(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        # Change directory to access onnx models
        abspath_this_file = os.path.abspath(inspect.getfile(self.__class__))
        static_cache_dir = os.path.dirname(abspath_this_file)
        os.chdir(static_cache_dir)
        os.chdir('../../files/models')

        self.extension = OnnxExtension()

    def test_deserialize_onnx_model(self):
        """ Function test_deserialize_sequential_with_defaults
        Test for Sequential Keras model deserialization
        Depends on correct implementation of model_to_flow

        :return: Nothing
        """

        task_lst = [4823, 52948]

        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                # Load the ONNX model from the file
                onnx_file = 'model_task_{}.onnx'.format(task.task_id)
                model_original = onnx.load(onnx_file)

                # Create an exact copy of the model
                onnx.save(model_original, 'model_copy.onnx')
                model_adjusted = onnx.load('model_copy.onnx')

                # Remove unneeded file
                os.remove('model_copy.onnx')

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
