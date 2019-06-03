import os
import sys
import inspect
import onnx
from google.protobuf import json_format

import openml
from openml.testing import TestBase
from openml.extensions.onnx import OnnxExtension

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)

# Values taken from https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3
# Might change in the future - changes here should be made in the extension and
# the serialization/deserialization unit tests as well
ONNX_ATTR_TYPES = {
    1: 'FLOAT',
    2: 'INT',
    3: 'STRING',
    4: 'TENSOR',
    5: 'GRAPH',
    6: 'FLOATS',
    7: 'INTS',
    8: 'STRINGS',
    9: 'TENSORS',
    10: 'GRAPHS'
}


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

    def _set_weigths_and_biases_to_zero(self, obj):
        if isinstance(obj, list):
            for item in obj:
                self._set_weigths_and_biases_to_zero(item)
        if isinstance(obj, dict):
            if 'dataType' in obj.keys():
                # Determine the name of the attribute containing the data
                if isinstance(obj['dataType'], int):
                    data_key = ONNX_ATTR_TYPES[obj['dataType']].lower() + 'Data'
                elif isinstance(obj['dataType'], str):
                    data_key = obj['dataType'].lower() + 'Data'
                else:
                    raise ValueError('Unknown data type. Try downgrading ONNX to 1.2.1.')

                for i in range(len(obj[data_key])):
                    obj[data_key][i] = 0.0
            else:
                for key, val in obj.items():
                    self._set_weigths_and_biases_to_zero(val)

    def test_deserialize_onnx_model(self):
        """ Function test_deserialize_onnx_model
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

                # Obtain the model's dictionary representations for comparison
                model_adj_dict = json_format.MessageToDict(model_adjusted)
                model_ori_dict = json_format.MessageToDict(model_original)

                # we want to confirm that model_original and model_adjusted are the same.
                self.assertDictEqual(model_adj_dict, model_ori_dict)

                flow = self.extension.model_to_flow(model_adjusted)
                model_deserialized = \
                    self.extension.flow_to_model(flow, initialize_with_defaults=True)

                # Obtain the dictionary representation of the deserialized model
                model_des_dict = json_format.MessageToDict(model_deserialized)

                # The weights and biases should be changed to zeros
                # as those values are removed during serialization
                self._set_weigths_and_biases_to_zero(model_ori_dict)

                # we want to confirm that model_original and model_deserialized are the same
                # so we compare their dictionaries.
                self.assertDictEqual(model_des_dict, model_ori_dict)
