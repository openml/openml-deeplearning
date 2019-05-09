import os
import sys
from unittest import mock
from collections import OrderedDict

import keras

from openml.extensions.keras import KerasExtension
from openml.testing import TestBase
from openml.datasets.functions import get_dataset
from openml import config

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestKerasExtensionFlowSerialization(TestBase):

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)

        self.extension = KerasExtension()

        config.server = self.production_server
        self.iris = get_dataset(61)
        self.mnist_784 = get_dataset(554)

        keras.backend.reset_uids()

    def test_serialize_sequential_model(self):
        with mock.patch.object(self.extension, '_check_dependencies') as check_dependencies_mock:
            model = keras.models.Sequential([
                keras.layers.BatchNormalization(),
                keras.layers.Dense(units=128, activation=keras.activations.relu),
                keras.layers.Dropout(rate=0.4),
                keras.layers.Dense(units=3, activation=keras.activations.softmax),
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            fixture_name = 'keras.engine.sequential.Sequential.ef8334f1d3c49c4f'
            fixture_description = 'Automatically created keras flow.'
            version_fixture = 'keras==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                              % keras.__version__
            fixture_parameters = \
                OrderedDict([('backend', '"tensorflow"'),
                             ('class_name', '"Sequential"'),
                             ('config', '{"name": "sequential_1"}'),
                             ('keras_version', '"2.2.4"'),
                             ('layer0_batch_normalization_1',
                              '{"class_name": "BatchNormalization", "config": {"axis": -1, '
                              '"beta_constraint": null, "beta_initializer": {"class_name": '
                              '"Zeros", "config": {}}, "beta_regularizer": null, "center": '
                              'true, "epsilon": 0.001, "gamma_constraint": null, '
                              '"gamma_initializer": {"class_name": "Ones", "config": {}}, '
                              '"gamma_regularizer": null, "momentum": 0.99, '
                              '"moving_mean_initializer": {"class_name": "Zeros", "config": '
                              '{}}, "moving_variance_initializer": {"class_name": "Ones", '
                              '"config": {}}, "name": "batch_normalization_1", "scale": true, '
                              '"trainable": true}}'),
                             ('layer1_dense_1',
                              '{"class_name": "Dense", "config": {"activation": "relu", '
                              '"activity_regularizer": null, "bias_constraint": null, '
                              '"bias_initializer": {"class_name": "Zeros", "config": {}}, '
                              '"bias_regularizer": null, "kernel_constraint": null, '
                              '"kernel_initializer": {"class_name": "VarianceScaling", '
                              '"config": {"distribution": "uniform", "mode": "fan_avg", '
                              '"scale": 1.0, "seed": null}}, "kernel_regularizer": null, '
                              '"name": "dense_1", "trainable": true, "units": 128, "use_bias": '
                              'true}}'),
                             ('layer2_dropout_1',
                              '{"class_name": "Dropout", "config": {"name": "dropout_1", '
                              '"noise_shape": null, "rate": 0.4, "seed": null, "trainable": '
                              'true}}'),
                             ('layer3_dense_2',
                              '{"class_name": "Dense", "config": {"activation": "softmax", '
                              '"activity_regularizer": null, "bias_constraint": null, '
                              '"bias_initializer": {"class_name": "Zeros", "config": {}}, '
                              '"bias_regularizer": null, "kernel_constraint": null, '
                              '"kernel_initializer": {"class_name": "VarianceScaling", '
                              '"config": {"distribution": "uniform", "mode": "fan_avg", '
                              '"scale": 1.0, "seed": null}}, "kernel_regularizer": null, '
                              '"name": "dense_2", "trainable": true, "units": 3, "use_bias": '
                              'true}}'),
                             ('optimizer',
                              '{"loss": "sparse_categorical_crossentropy", "loss_weights": '
                              'null, "metrics": ["accuracy"], "optimizer_config": '
                              '{"class_name": "Adam", "config": {"amsgrad": false, "beta_1": '
                              '0.8999999761581421, "beta_2": 0.9990000128746033, "decay": 0.0, '
                              '"epsilon": 1e-07, "lr": 0.0010000000474974513}}, '
                              '"sample_weight_mode": null, "weighted_metrics": null}')])
            structure_fixture = {'keras.engine.sequential.Sequential.ef8334f1d3c49c4f': []}

            serialization = self.extension.model_to_flow(model)
            structure = serialization.get_structure('name')

            self.assertEqual(serialization.name, fixture_name)
            self.assertEqual(serialization.class_name, fixture_name)
            self.assertEqual(serialization.description, fixture_description)
            self.assertEqual(serialization.parameters, fixture_parameters)
            self.assertEqual(serialization.dependencies, version_fixture)
            self.assertDictEqual(structure, structure_fixture)

            new_model = self.extension.flow_to_model(serialization)
            # compares string representations of the dict, as it potentially
            # contains complex objects that can not be compared with == op
            self.assertEqual(str(model.get_config()), str(new_model.get_config()))

            self.assertEqual(type(new_model), type(model))
            self.assertIsNot(new_model, model)

            self.assertEqual(new_model.get_config(), model.get_config())

            X, y, _, _ = self.iris.get_data(
                dataset_format='array',
                target=self.iris.default_target_attribute
            )
            new_model.fit(X, y)

            self.assertEqual(check_dependencies_mock.call_count, 1)

    def test_serialize_functional_model(self):
        with mock.patch.object(self.extension, '_check_dependencies') as check_dependencies_mock:
            inp = keras.layers.Input(shape=(784,))
            normalized = keras.layers.BatchNormalization()(inp)
            dense = keras.layers.Dense(units=64, activation='relu')(normalized)
            dropout = keras.layers.Dropout(rate=0.4)(dense)
            out = keras.layers.Dense(units=10, activation='softmax')(dropout)
            model = keras.models.Model(inp, out)

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            fixture_name = 'keras.engine.training.Model.e070b50a21e68237'
            fixture_description = 'Automatically created keras flow.'
            version_fixture = 'keras==%s\nnumpy>=1.6.1\nscipy>=0.9' \
                              % keras.__version__
            fixture_parameters = \
                OrderedDict([('backend', '"tensorflow"'),
                             ('class_name', '"Model"'),
                             ('config',
                              '{"input_layers": [["input_1", 0, 0]], "name": "model_1", '
                              '"output_layers": [["dense_2", 0, 0]]}'),
                             ('keras_version', '"2.2.4"'),
                             ('layer0_input_1',
                              '{"class_name": "InputLayer", "config": {"batch_input_shape": '
                              '[null, 784], "dtype": "float32", "name": "input_1", "sparse": '
                              'false}, "inbound_nodes": [], "name": "input_1"}'),
                             ('layer1_batch_normalization_1',
                              '{"class_name": "BatchNormalization", "config": {"axis": -1, '
                              '"beta_constraint": null, "beta_initializer": {"class_name": '
                              '"Zeros", "config": {}}, "beta_regularizer": null, "center": '
                              'true, "epsilon": 0.001, "gamma_constraint": null, '
                              '"gamma_initializer": {"class_name": "Ones", "config": {}}, '
                              '"gamma_regularizer": null, "momentum": 0.99, '
                              '"moving_mean_initializer": {"class_name": "Zeros", "config": '
                              '{}}, "moving_variance_initializer": {"class_name": "Ones", '
                              '"config": {}}, "name": "batch_normalization_1", "scale": true, '
                              '"trainable": true}, "inbound_nodes": [[["input_1", 0, 0, {}]]], '
                              '"name": "batch_normalization_1"}'),
                             ('layer2_dense_1',
                              '{"class_name": "Dense", "config": {"activation": "relu", '
                              '"activity_regularizer": null, "bias_constraint": null, '
                              '"bias_initializer": {"class_name": "Zeros", "config": {}}, '
                              '"bias_regularizer": null, "kernel_constraint": null, '
                              '"kernel_initializer": {"class_name": "VarianceScaling", '
                              '"config": {"distribution": "uniform", "mode": "fan_avg", '
                              '"scale": 1.0, "seed": null}}, "kernel_regularizer": null, '
                              '"name": "dense_1", "trainable": true, "units": 64, "use_bias": '
                              'true}, "inbound_nodes": [[["batch_normalization_1", 0, 0, '
                              '{}]]], "name": "dense_1"}'),
                             ('layer3_dropout_1',
                              '{"class_name": "Dropout", "config": {"name": "dropout_1", '
                              '"noise_shape": null, "rate": 0.4, "seed": null, "trainable": '
                              'true}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "name": '
                              '"dropout_1"}'),
                             ('layer4_dense_2',
                              '{"class_name": "Dense", "config": {"activation": "softmax", '
                              '"activity_regularizer": null, "bias_constraint": null, '
                              '"bias_initializer": {"class_name": "Zeros", "config": {}}, '
                              '"bias_regularizer": null, "kernel_constraint": null, '
                              '"kernel_initializer": {"class_name": "VarianceScaling", '
                              '"config": {"distribution": "uniform", "mode": "fan_avg", '
                              '"scale": 1.0, "seed": null}}, "kernel_regularizer": null, '
                              '"name": "dense_2", "trainable": true, "units": 10, "use_bias": '
                              'true}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "name": '
                              '"dense_2"}'),
                             ('optimizer',
                              '{"loss": "sparse_categorical_crossentropy", "loss_weights": '
                              'null, "metrics": ["accuracy"], "optimizer_config": '
                              '{"class_name": "Adam", "config": {"amsgrad": false, "beta_1": '
                              '0.8999999761581421, "beta_2": 0.9990000128746033, "decay": 0.0, '
                              '"epsilon": 1e-07, "lr": 0.0010000000474974513}}, '
                              '"sample_weight_mode": null, "weighted_metrics": null}')])
            structure_fixture = {'keras.engine.training.Model.e070b50a21e68237': []}

            serialization = self.extension.model_to_flow(model)
            structure = serialization.get_structure('name')

            self.assertEqual(serialization.name, fixture_name)
            self.assertEqual(serialization.class_name, fixture_name)
            self.assertEqual(serialization.description, fixture_description)
            self.assertEqual(serialization.parameters, fixture_parameters)
            self.assertEqual(serialization.dependencies, version_fixture)
            self.assertDictEqual(structure, structure_fixture)

            new_model = self.extension.flow_to_model(serialization)
            # compares string representations of the dict, as it potentially
            # contains complex objects that can not be compared with == op
            self.assertEqual(str(model.get_config()), str(new_model.get_config()))

            self.assertEqual(type(new_model), type(model))
            self.assertIsNot(new_model, model)

            self.assertEqual(new_model.get_config(), model.get_config())

            X, y, _, _ = self.mnist_784.get_data(
                dataset_format='array',
                target=self.mnist_784.default_target_attribute
            )
            new_model.fit(X, y)

            self.assertEqual(check_dependencies_mock.call_count, 1)
