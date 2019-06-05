import os
import sys
from collections import OrderedDict
from unittest import mock

import mxnet as mx
from mxnet.gluon import nn

from openml import config
from openml.extensions.mxnet import MXNetExtension
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestMXNetExtensionFlowSerialization(TestBase):

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)
        self.extension = MXNetExtension()
        config.server = self.production_server

        mx.name.NameManager._current.value = mx.name.NameManager()

    def test_serialize_sequential_model(self):
        with mock.patch.object(self.extension, '_check_dependencies') as check_dependencies_mock:
            model = nn.HybridSequential()
            with model.name_scope():
                model.add(
                    nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                    nn.MaxPool2D(pool_size=2, strides=2),
                    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                    nn.MaxPool2D(pool_size=2, strides=2),
                    nn.Dense(120, activation="relu"),
                    nn.Dense(84, activation="relu"),
                    nn.Dense(10)
                )
            model.collect_params().initialize(mx.init.Constant(0))

            fixture_name = 'mxnet.gluon.nn.basic_layers.HybridSequential'
            fixture_description = 'Automatically created MXNet flow.'
            version_fixture = 'mxnet==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                              % mx.__version__
            fixture_parameters = \
                OrderedDict([('00_data', '{"inputs": [], "name": "data", "op": "null"}'),
                             ('01_hybridsequential0_conv0_weight',
                              '{"attrs": {"__dtype__": "0", "__lr_mult__": "1.0", "__shape__": '
                              '"(6, 0, 5, 5)", "__storage_type__": "0", "__wd_mult__": "1.0"}, '
                              '"inputs": [], "name": "hybridsequential0_conv0_weight", "op": '
                              '"null"}'),
                             ('02_hybridsequential0_conv0_bias',
                              '{"attrs": {"__dtype__": "0", "__init__": "zeros", '
                              '"__lr_mult__": "1.0", "__shape__": "(6,)", "__storage_type__": '
                              '"0", "__wd_mult__": "1.0"}, "inputs": [], "name": '
                              '"hybridsequential0_conv0_bias", "op": "null"}'),
                             ('03_hybridsequential0_conv0_fwd',
                              '{"attrs": {"dilate": "(1, 1)", "kernel": "(5, 5)", "layout": '
                              '"NCHW", "no_bias": "False", "num_filter": "6", "num_group": '
                              '"1", "pad": "(0, 0)", "stride": "(1, 1)"}, "inputs": [[0, 0, '
                              '0], [1, 0, 0], [2, 0, 0]], "name": '
                              '"hybridsequential0_conv0_fwd", "op": "Convolution"}'),
                             ('04_hybridsequential0_conv0_relu_fwd',
                              '{"attrs": {"act_type": "relu"}, "inputs": [[3, 0, 0]], "name": '
                              '"hybridsequential0_conv0_relu_fwd", "op": "Activation"}'),
                             ('05_hybridsequential0_pool0_fwd',
                              '{"attrs": {"global_pool": "False", "kernel": "(2, 2)", "pad": '
                              '"(0, 0)", "pool_type": "max", "pooling_convention": "valid", '
                              '"stride": "(2, 2)"}, "inputs": [[4, 0, 0]], "name": '
                              '"hybridsequential0_pool0_fwd", "op": "Pooling"}'),
                             ('06_hybridsequential0_conv1_weight',
                              '{"attrs": {"__dtype__": "0", "__lr_mult__": "1.0", "__shape__": '
                              '"(16, 0, 3, 3)", "__storage_type__": "0", "__wd_mult__": '
                              '"1.0"}, "inputs": [], "name": "hybridsequential0_conv1_weight", '
                              '"op": "null"}'),
                             ('07_hybridsequential0_conv1_bias',
                              '{"attrs": {"__dtype__": "0", "__init__": "zeros", '
                              '"__lr_mult__": "1.0", "__shape__": "(16,)", "__storage_type__": '
                              '"0", "__wd_mult__": "1.0"}, "inputs": [], "name": '
                              '"hybridsequential0_conv1_bias", "op": "null"}'),
                             ('08_hybridsequential0_conv1_fwd',
                              '{"attrs": {"dilate": "(1, 1)", "kernel": "(3, 3)", "layout": '
                              '"NCHW", "no_bias": "False", "num_filter": "16", "num_group": '
                              '"1", "pad": "(0, 0)", "stride": "(1, 1)"}, "inputs": [[5, 0, '
                              '0], [6, 0, 0], [7, 0, 0]], "name": '
                              '"hybridsequential0_conv1_fwd", "op": "Convolution"}'),
                             ('09_hybridsequential0_conv1_relu_fwd',
                              '{"attrs": {"act_type": "relu"}, "inputs": [[8, 0, 0]], "name": '
                              '"hybridsequential0_conv1_relu_fwd", "op": "Activation"}'),
                             ('10_hybridsequential0_pool1_fwd',
                              '{"attrs": {"global_pool": "False", "kernel": "(2, 2)", "pad": '
                              '"(0, 0)", "pool_type": "max", "pooling_convention": "valid", '
                              '"stride": "(2, 2)"}, "inputs": [[9, 0, 0]], "name": '
                              '"hybridsequential0_pool1_fwd", "op": "Pooling"}'),
                             ('11_hybridsequential0_dense0_weight',
                              '{"attrs": {"__dtype__": "0", "__lr_mult__": "1.0", "__shape__": '
                              '"(120, 0)", "__storage_type__": "0", "__wd_mult__": "1.0"}, '
                              '"inputs": [], "name": "hybridsequential0_dense0_weight", "op": '
                              '"null"}'),
                             ('12_hybridsequential0_dense0_bias',
                              '{"attrs": {"__dtype__": "0", "__init__": "zeros", '
                              '"__lr_mult__": "1.0", "__shape__": "(120,)", '
                              '"__storage_type__": "0", "__wd_mult__": "1.0"}, "inputs": [], '
                              '"name": "hybridsequential0_dense0_bias", "op": "null"}'),
                             ('13_hybridsequential0_dense0_fwd',
                              '{"attrs": {"flatten": "True", "no_bias": "False", "num_hidden": '
                              '"120"}, "inputs": [[10, 0, 0], [11, 0, 0], [12, 0, 0]], "name": '
                              '"hybridsequential0_dense0_fwd", "op": "FullyConnected"}'),
                             ('14_hybridsequential0_dense0_relu_fwd',
                              '{"attrs": {"act_type": "relu"}, "inputs": [[13, 0, 0]], "name": '
                              '"hybridsequential0_dense0_relu_fwd", "op": "Activation"}'),
                             ('15_hybridsequential0_dense1_weight',
                              '{"attrs": {"__dtype__": "0", "__lr_mult__": "1.0", "__shape__": '
                              '"(84, 0)", "__storage_type__": "0", "__wd_mult__": "1.0"}, '
                              '"inputs": [], "name": "hybridsequential0_dense1_weight", "op": '
                              '"null"}'),
                             ('16_hybridsequential0_dense1_bias',
                              '{"attrs": {"__dtype__": "0", "__init__": "zeros", '
                              '"__lr_mult__": "1.0", "__shape__": "(84,)", "__storage_type__": '
                              '"0", "__wd_mult__": "1.0"}, "inputs": [], "name": '
                              '"hybridsequential0_dense1_bias", "op": "null"}'),
                             ('17_hybridsequential0_dense1_fwd',
                              '{"attrs": {"flatten": "True", "no_bias": "False", "num_hidden": '
                              '"84"}, "inputs": [[14, 0, 0], [15, 0, 0], [16, 0, 0]], "name": '
                              '"hybridsequential0_dense1_fwd", "op": "FullyConnected"}'),
                             ('18_hybridsequential0_dense1_relu_fwd',
                              '{"attrs": {"act_type": "relu"}, "inputs": [[17, 0, 0]], "name": '
                              '"hybridsequential0_dense1_relu_fwd", "op": "Activation"}'),
                             ('19_hybridsequential0_dense2_weight',
                              '{"attrs": {"__dtype__": "0", "__lr_mult__": "1.0", "__shape__": '
                              '"(10, 0)", "__storage_type__": "0", "__wd_mult__": "1.0"}, '
                              '"inputs": [], "name": "hybridsequential0_dense2_weight", "op": '
                              '"null"}'),
                             ('20_hybridsequential0_dense2_bias',
                              '{"attrs": {"__dtype__": "0", "__init__": "zeros", '
                              '"__lr_mult__": "1.0", "__shape__": "(10,)", "__storage_type__": '
                              '"0", "__wd_mult__": "1.0"}, "inputs": [], "name": '
                              '"hybridsequential0_dense2_bias", "op": "null"}'),
                             ('21_hybridsequential0_dense2_fwd',
                              '{"attrs": {"flatten": "True", "no_bias": "False", "num_hidden": '
                              '"10"}, "inputs": [[18, 0, 0], [19, 0, 0], [20, 0, 0]], "name": '
                              '"hybridsequential0_dense2_fwd", "op": "FullyConnected"}'),
                             ('misc',
                              '{"arg_nodes": [0, 1, 2, 6, 7, 11, 12, 15, 16, 19, 20], "attrs": '
                              '{"mxnet_version": ["int", 10401]}, "heads": [[21, 0, 0]], '
                              '"node_row_ptr": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, '
                              '14, 15, 16, 17, 18, 19, 20, 21, 22]}')])

            serialization = self.extension.model_to_flow(model)

            self.assertIn(fixture_name, serialization.name)
            self.assertEqual(serialization.class_name[:len(fixture_name)], fixture_name)
            self.assertEqual(serialization.description, fixture_description)
            self.assertEqual(serialization.parameters, fixture_parameters)
            self.assertEqual(serialization.dependencies, version_fixture)

            self.assertEqual(check_dependencies_mock.call_count, 0)
