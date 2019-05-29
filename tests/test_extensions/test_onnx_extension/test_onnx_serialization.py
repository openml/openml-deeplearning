import os
import sys
import onnx
import inspect
from collections import OrderedDict

import openml
from openml.extensions.onnx import OnnxExtension
from openml.testing import TestBase


this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestOnnxExtensionSerialization(TestBase):

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        # Change directory to access onnx models
        abspath_this_file = os.path.abspath(inspect.getfile(self.__class__))
        static_cache_dir = os.path.dirname(abspath_this_file)
        os.chdir(static_cache_dir)
        os.chdir('../../files/models')

        self.extension = OnnxExtension()

    def test_serialize_onnx_model(self):
        model_mx = onnx.load('model_task_10101.onnx')
        flow = self.extension.model_to_flow(model_mx)

        # Create the fixed values to assert against
        fixed_description = 'Automatically created ONNX flow.'
        fixed_version = 'onnx==1.2.1,openml==0.8.0'
        fixed_dependencies = 'onnx==1.2.1\nmxnet==1.4.1\nnumpy>=1.6.1\nscipy>=1.2.1'
        fixed_params = \
            OrderedDict([('backend', '{"irVersion": "3", "opsetImport": [{"version": "7"}]}'),
                         ('initializer_0_batchnorm0_gamma',
                          '{"dataType": "FLOAT", "dims": ["4"], "name": "batchnorm0_gamma"}'),
                         ('initializer_1_batchnorm0_beta',
                          '{"dataType": "FLOAT", "dims": ["4"], "name": "batchnorm0_beta"}'),
                         ('initializer_2_batchnorm0_moving_mean',
                          '{"dataType": "FLOAT", "dims": ["4"], "name": "batchnorm0_moving_mean"}'),
                         ('initializer_3_batchnorm0_moving_var',
                          '{"dataType": "FLOAT", "dims": ["4"], "name": "batchnorm0_moving_var"}'),
                         ('initializer_4_fullyconnected0_weight',
                          '{"dataType": "FLOAT", "dims": ["1024", "4"],'
                          ' "name": "fullyconnected0_weight"}'),
                         ('initializer_5_fullyconnected0_bias',
                          '{"dataType": "FLOAT", "dims": ["1024"],'
                          ' "name": "fullyconnected0_bias"}'),
                         ('initializer_6_fullyconnected1_weight',
                          '{"dataType": "FLOAT", "dims": ["1024", "1024"], '
                          '"name": "fullyconnected1_weight"}'),
                         ('initializer_7_fullyconnected1_bias',
                          '{"dataType": "FLOAT", "dims": ["1024"], '
                          '"name": "fullyconnected1_bias"}'),
                         ('initializer_8_fullyconnected2_weight',
                          '{"dataType": "FLOAT", "dims": ["2", "1024"], '
                          '"name": "fullyconnected2_weight"}'),
                         ('initializer_9_fullyconnected2_bias',
                          '{"dataType": "FLOAT", "dims": ["2"], "name": "fullyconnected2_bias"}'),
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
                         ('node_0_batchnorm0',
                          '{"attribute": [{"name": "epsilon", "f": 0.0010000000474974513, '
                          '"type": "FLOAT"}, {"name": "momentum", "f": 0.8999999761581421, '
                          '"type": "FLOAT"}, {"name": "spatial", "i": "0", "type": "INT"}], '
                          '"input": ["data", "batchnorm0_gamma", "batchnorm0_beta", '
                          '"batchnorm0_moving_mean", "batchnorm0_moving_var"], '
                          '"name": "batchnorm0", "opType": "BatchNormalization", '
                          '"output": ["batchnorm0"]}'),
                         ('node_1_fullyconnected0',
                          '{"attribute": [{"name": "alpha", "f": 1.0, "type": "FLOAT"}, '
                          '{"name": "beta", "f": 1.0, "type": "FLOAT"}, '
                          '{"name": "transA", "i": "0", "type": "INT"}, '
                          '{"name": "transB", "i": "1", "type": "INT"}], '
                          '"input": ["batchnorm0", "fullyconnected0_weight", '
                          '"fullyconnected0_bias"], "name": "fullyconnected0", '
                          '"opType": "Gemm", "output": ["fullyconnected0"]}'),
                         ('node_2_activation0',
                          '{"input": ["fullyconnected0"], "name": "activation0", "opType": "Relu", '
                          '"output": ["activation0"]}'),
                         ('node_3_dropout0',
                          '{"attribute": [{"name": "ratio", "f": 0.10000000149011612, '
                          '"type": "FLOAT"}], "input": ["activation0"], "name": "dropout0", '
                          '"opType": "Dropout", "output": ["dropout0"]}'),
                         ('node_4_fullyconnected1',
                          '{"attribute": [{"name": "alpha", "f": 1.0, "type": "FLOAT"}, '
                          '{"name": "beta", "f": 1.0, "type": "FLOAT"}, {"name": "transA", '
                          '"i": "0", "type": "INT"}, {"name": "transB", "i": "1", "type": "INT"}], '
                          '"input": ["dropout0", "fullyconnected1_weight", "fullyconnected1_bias"],'
                          ' "name": "fullyconnected1", "opType": "Gemm", '
                          '"output": ["fullyconnected1"]}'),
                         ('node_5_activation1',
                          '{"input": ["fullyconnected1"], "name": "activation1", '
                          '"opType": "Relu", "output": ["activation1"]}'),
                         ('node_6_dropout1',
                          '{"attribute": [{"name": "ratio", "f": 0.20000000298023224, '
                          '"type": "FLOAT"}], "input": ["activation1"], "name": "dropout1", '
                          '"opType": "Dropout", "output": ["dropout1"]}'),
                         ('node_7_fullyconnected2',
                          '{"attribute": [{"name": "alpha", "f": 1.0, "type": "FLOAT"}, '
                          '{"name": "beta", "f": 1.0, "type": "FLOAT"}, {"name": "transA", '
                          '"i": "0", "type": "INT"}, {"name": "transB", "i": "1", "type": "INT"}], '
                          '"input": ["dropout1", "fullyconnected2_weight", "fullyconnected2_bias"],'
                          ' "name": "fullyconnected2", "opType": "Gemm", '
                          '"output": ["fullyconnected2"]}'),
                         ('node_8_softmax',
                          '{"attribute": [{"name": "axis", "i": "1", "type": "INT"}], '
                          '"input": ["fullyconnected2"], "name": "softmax", '
                          '"opType": "Softmax", "output": ["softmax"]}'),
                         ('output_0_softmax',
                          '{"name": "softmax", "type": {"tensorType": {"elemType": "FLOAT", '
                          '"shape": {"dim": [{"dimValue": "1024"}, {"dimValue": "2"}]}}}}')])

        # The tests itself
        self.assertEqual(flow.description, fixed_description)
        self.assertDictEqual(flow.parameters, fixed_params)
        self.assertEqual(flow.dependencies, fixed_dependencies)
        self.assertEqual(flow.external_version, fixed_version)
