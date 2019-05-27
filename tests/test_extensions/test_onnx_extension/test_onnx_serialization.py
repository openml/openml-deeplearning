import os
import sys
import onnx
import numpy as np
from collections import OrderedDict

from .onnx_model_utils import create_onnx_file, remove_onnx_file, remove_mxnet_files

import openml
from openml.extensions.onnx import OnnxExtension
from openml.testing import TestBase


this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestONNXExtensionSerialization(TestBase):

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        self.extension = OnnxExtension()

    def tearDown(self):
        remove_mxnet_files()
        remove_onnx_file()

    def test_serialize_onnx_model(self):

        # Get the task and split the data
        task = openml.tasks.get_task(10101)
        X, y = task.get_X_and_y()
        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=0, fold=0, sample=0)
        X_train = X[train_indices]
        X_test = X[test_indices]

        # Sanitize null values
        X_train[np.isnan(X_train)] = 1.0e-12
        X_test[np.isnan(X_test)] = 1.0e-12

        output_length = len(task.class_labels)
        input_length = X_train.shape[1]

        # Create the pre-made model, serialize and delete the file
        create_onnx_file(input_length, output_length, X_train, task)
        model_mx = onnx.load('model.onnx')
        flow = self.extension.model_to_flow(model_mx)
        remove_onnx_file()

        # Create the fixed values to assert against
        fixed_name = 'onnx_ml_pb2.ModelProto.fc21d829'
        fixed_description = 'Automatically created ONNX flow.'
        fixed_version = 'onnx==1.2.1,openml==0.8.0'
        fixed_dependencies = 'onnx==1.2.1\nmxnet==1.4.1\nnumpy==1.14.6\nscipy==1.2.1'
        fixed_params = \
            OrderedDict([('backend', '{"irVersion": "3", "opsetImport": [{"version": "7"}]}'),
                         ('initializer_0_batchnorm0_gamma',
                          '{"dims": ["4"], "dataType": "FLOAT", "name": "batchnorm0_gamma"}'),
                         ('initializer_1_batchnorm0_beta',
                          '{"dims": ["4"], "dataType": "FLOAT", "name": "batchnorm0_beta"}'),
                         ('initializer_2_batchnorm0_moving_mean',
                          '{"dims": ["4"], "dataType": "FLOAT", "name": "batchnorm0_moving_mean"}'),
                         ('initializer_3_batchnorm0_moving_var',
                          '{"dims": ["4"], "dataType": "FLOAT", "name": "batchnorm0_moving_var"}'),
                         ('initializer_4_fullyconnected0_weight',
                          '{"dims": ["1024", "4"], "dataType": "FLOAT",'
                          ' "name": "fullyconnected0_weight"}'),
                         ('initializer_5_fullyconnected0_bias',
                          '{"dims": ["1024"], "dataType": "FLOAT",'
                          ' "name": "fullyconnected0_bias"}'),
                         ('initializer_6_fullyconnected1_weight',
                          '{"dims": ["1024", "1024"], "dataType":'
                          ' "FLOAT", "name": "fullyconnected1_weight"}'),
                         ('initializer_7_fullyconnected1_bias',
                          '{"dims": ["1024"], "dataType":'
                          ' "FLOAT", "name": "fullyconnected1_bias"}'),
                         ('initializer_8_fullyconnected2_weight',
                          '{"dims": ["2", "1024"], "dataType":'
                          ' "FLOAT", "name": "fullyconnected2_weight"}'),
                         ('initializer_9_fullyconnected2_bias',
                          '{"dims": ["2"], "dataType": "FLOAT", "name": "fullyconnected2_bias"}'),
                         ('input_0_data',
                          '{"name": "data", "type":'
                          ' {"tensorType": {"elemType": "FLOAT", "shape": {"dim":'
                          ' [{"dimValue": "1024"}, {"dimValue": "4"}]}}}}'),
                         ('input_10_fullyconnected2_bias',
                          '{"name": "fullyconnected2_bias",'
                          ' "type": {"tensorType": {"elemType": "FLOAT",'
                          ' "shape": {"dim": [{"dimValue": "2"}]}}}}'),
                         ('input_1_batchnorm0_gamma',
                          '{"name": "batchnorm0_gamma", "type": {"tensorType": {"elemType": "FLOAT"'
                          ', "shape": {"dim": [{"dimValue": "4"}]}}}}'),
                         ('input_2_batchnorm0_beta',
                          '{"name": "batchnorm0_beta", "type": {"tensorType": {"elemType": "FLOAT",'
                          ' "shape": {"dim": [{"dimValue": "4"}]}}}}'),
                         ('input_3_batchnorm0_moving_mean',
                          '{"name": "batchnorm0_moving_mean", "type": {"tensorType": {"elemType":'
                          ' "FLOAT", "shape": {"dim": [{"dimValue": "4"}]}}}}'),
                         ('input_4_batchnorm0_moving_var',
                          '{"name": "batchnorm0_moving_var", "type": {"tensorType": {"elemType":'
                          ' "FLOAT", "shape": {"dim": [{"dimValue": "4"}]}}}}'),
                         ('input_5_fullyconnected0_weight',
                          '{"name": "fullyconnected0_weight", "type": {"tensorType": {"elemType":'
                          ' "FLOAT", "shape": {"dim": [{"dimValue": "1024"},'
                          ' {"dimValue": "4"}]}}}}'),
                         ('input_6_fullyconnected0_bias',
                          '{"name": "fullyconnected0_bias", "type": {"tensorType":'
                          ' {"elemType": "FLOAT", "shape": {"dim": [{"dimValue": "1024"}]}}}}'),
                         ('input_7_fullyconnected1_weight',
                          '{"name": "fullyconnected1_weight", "type": {"tensorType":'
                          ' {"elemType": "FLOAT", "shape": {"dim":'
                          ' [{"dimValue": "1024"}, {"dimValue": "1024"}]}}}}'),
                         ('input_8_fullyconnected1_bias',
                          '{"name": "fullyconnected1_bias", "type": {"tensorType":'
                          ' {"elemType": "FLOAT", "shape": {"dim": [{"dimValue": "1024"}]}}}}'),
                         ('input_9_fullyconnected2_weight',
                          '{"name": "fullyconnected2_weight", "type":'
                          ' {"tensorType": {"elemType": "FLOAT", "shape":'
                          ' {"dim": [{"dimValue": "2"}, {"dimValue": "1024"}]}}}}'),
                         ('name', 'mxnet_converted_model'),
                         ('node_0_batchnorm0', '{"input": ["data", "batchnorm0_gamma",'
                                               ' "batchnorm0_beta", "batchnorm0_moving_mean",'
                                               ' "batchnorm0_moving_var"],'
                                               ' "output": ["batchnorm0"],'
                                               ' "name": "batchnorm0", "opType":'
                                               ' "BatchNormalization", "attribute":'
                                               ' [{"name": "epsilon", "f": 0.0010000000474974513,'
                                               ' "type": "FLOAT"}, {"name": "momentum",'
                                               ' "f": 0.8999999761581421, "type": "FLOAT"},'
                                               ' {"name": "spatial", "i": "0", "type": "INT"}]}'),
                         ('node_1_fullyconnected0',
                          '{"input": ["batchnorm0", "fullyconnected0_weight",'
                          ' "fullyconnected0_bias"], "output": ["fullyconnected0"],'
                          ' "name": "fullyconnected0", "opType": "Gemm", "attribute":'
                          ' [{"name": "alpha", "f": 1.0, "type": "FLOAT"}, {"name": "beta",'
                          ' "f": 1.0, "type": "FLOAT"}, {"name": "transA", "i": "0", "type":'
                          ' "INT"}, {"name": "transB", "i": "1", "type": "INT"}]}'),
                         ('node_2_activation0',
                          '{"input": ["fullyconnected0"], "output": ["activation0"], "name":'
                          ' "activation0", "opType": "Relu"}'),
                         ('node_3_dropout0',
                          '{"input": ["activation0"], "output": ["dropout0"], "name": "dropout0",'
                          ' "opType": "Dropout", "attribute": [{"name":'
                          ' "ratio", "f": 0.10000000149011612, "type": "FLOAT"}]}'),
                         ('node_4_fullyconnected1',
                          '{"input": ["dropout0", "fullyconnected1_weight",'
                          ' "fullyconnected1_bias"], "output": ["fullyconnected1"],'
                          ' "name": "fullyconnected1", "opType": "Gemm", "attribute":'
                          ' [{"name": "alpha", "f": 1.0, "type": "FLOAT"},'
                          ' {"name": "beta", "f": 1.0, "type": "FLOAT"},'
                          ' {"name": "transA", "i": "0", "type": "INT"},'
                          ' {"name": "transB", "i": "1", "type": "INT"}]}'),
                         ('node_5_activation1',
                          '{"input": ["fullyconnected1"], "output":'
                          ' ["activation1"], "name": "activation1", "opType": "Relu"}'),
                         ('node_6_dropout1',
                          '{"input": ["activation1"], "output":'
                          ' ["dropout1"], "name": "dropout1", "opType": "Dropout", "attribute":'
                          ' [{"name": "ratio", "f": 0.20000000298023224, "type": "FLOAT"}]}'),
                         ('node_7_fullyconnected2',
                          '{"input": ["dropout1", "fullyconnected2_weight",'
                          ' "fullyconnected2_bias"], "output": ["fullyconnected2"],'
                          ' "name": "fullyconnected2", "opType": "Gemm", "attribute":'
                          ' [{"name": "alpha", "f": 1.0, "type": "FLOAT"},'
                          ' {"name": "beta", "f": 1.0, "type": "FLOAT"}, {"name": "transA", "i":'
                          ' "0", "type": "INT"}, {"name": "transB", "i": "1", "type": "INT"}]}'),
                         ('node_8_softmax',
                          '{"input": ["fullyconnected2"], "output": ["softmax"], "name": "softmax",'
                          ' "opType": "Softmax", "attribute":'
                          ' [{"name": "axis", "i": "1", "type": "INT"}]}'),
                         ('output_0_softmax',
                          '{"name": "softmax", "type": {"tensorType": {"elemType":'
                          ' "FLOAT", "shape": {"dim": [{"dimValue": "1024"},'
                          ' {"dimValue": "2"}]}}}}')])

        # The tests itself
        self.assertEqual(flow.name, fixed_name)
        self.assertEqual(flow.class_name, fixed_name)
        self.assertEqual(flow.description, fixed_description)
        self.assertEqual(flow.parameters, fixed_params)
        self.assertEqual(flow.dependencies, fixed_dependencies)
        self.assertEqual(flow.external_version, fixed_version)
