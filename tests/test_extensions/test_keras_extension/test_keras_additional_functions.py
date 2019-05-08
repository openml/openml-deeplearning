import unittest
from distutils.version import LooseVersion
from unittest import mock

import keras
import sklearn

from sklearn import pipeline
from sklearn import tree

if LooseVersion(sklearn.__version__) < "0.20":
    from sklearn.preprocessing import Imputer
else:
    from sklearn.impute import SimpleImputer as Imputer

from openml.extensions.keras import KerasExtension
from openml.extensions.sklearn import SklearnExtension

from openml.setups.setup import OpenMLParameter

class SklearnModel(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, X, y):
        pass


class KerasModel(keras.models.Model):

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        pass

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        pass


class TestKerasExtensionAdditionalFunctions(unittest.TestCase):

    def setUp(self):
        self.sklearnModel = SklearnModel('true', '1', '0.1')
        self.kerasModel = KerasModel()

        self.extension = KerasExtension()

        self.keras_dummy_model = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=256, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(units=2, activation=keras.activations.softmax),
        ])

    def test_can_handle_model(self):
        is_keras = self.extension.can_handle_model(self.kerasModel)
        is_not_keras = self.extension.can_handle_model(self.sklearnModel)

        self.assertTrue(is_keras)
        self.assertFalse(is_not_keras)

    def test_get_version_information(self):
        self.version_information = self.extension.get_version_information()

        self.assertIsInstance(self.version_information, list)
        self.assertIsNotNone(self.version_information)

    def test_create_setup_string(self):
        self.setupStringResult = self.extension.create_setup_string(self.kerasModel)
        self.assertIsInstance(self.setupStringResult, str)

    def test_is_estimator(self):
        is_estimator = self.extension.is_estimator(self.kerasModel)
        is_not_estimator = self.extension.is_estimator(self.sklearnModel)

        self.assertTrue(is_estimator)
        self.assertFalse(is_not_estimator)

    def test__is_keras_flow(self):
        self.sklearn_dummy_model = pipeline.Pipeline(
            steps=[
                ('imputer', Imputer()),
                ('estimator', tree.DecisionTreeClassifier())
            ]
        )

        self.keras_flow = self.extension.model_to_flow(self.keras_dummy_model)
        self.keras_flow_external_version = self.extension._is_keras_flow(self.keras_flow)

        self.sklearn_flow = SklearnExtension().model_to_flow(self.sklearn_dummy_model)
        self.sklearn_flow_external_version = self.extension._is_keras_flow(self.sklearn_flow)

        self.assertTrue(self.keras_flow_external_version)
        self.assertFalse(self.sklearn_flow_external_version)

    @mock.patch('warnings.warn')
    def test__check_dependencies(self, warnings_mock):
        dependencies = ['keras==0.1', 'keras>=99.99.99',
                        'keras>99.99.99']
        for dependency in dependencies:
            self.assertRaises(ValueError, self.extension._check_dependencies, dependency)

    def test__openml_param_name_to_keras(self):
        openml_param_dummy = OpenMLParameter(1, 9763, " keras.engine.sequential.Sequential.B5679DFA264EC778",
                                       "keras.engine.sequential.Sequential.B5679DFA264EC778",
                                       "backend", "", "", "")

        self.keras_flow = self.extension.model_to_flow(self.keras_dummy_model)

        # Check if exception is thrown when OpenMLParam and Flow do not correspond
        self.assertRaises(ValueError, self.extension._openml_param_name_to_keras, openml_param_dummy, self.keras_flow)


if __name__ == '__main__':
    unittest.main()
