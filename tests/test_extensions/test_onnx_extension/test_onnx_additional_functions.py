import os
import sys
import numpy as np

import openml
from openml.extensions.onnx import OnnxExtension
from openml import config
from openml.testing import TestBase

import onnx
from .onnx_model_utils import create_onnx_file, remove_onnx_file

from collections import OrderedDict

import sklearn
from openml.extensions.sklearn import SklearnExtension
from sklearn import pipeline
from sklearn import tree

from distutils.version import LooseVersion

if LooseVersion(sklearn.__version__) < "0.20":
    from sklearn.preprocessing import Imputer
else:
    from sklearn.impute import SimpleImputer as Imputer

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class SklearnModel(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, x, y):
        pass


class TestONNXExtensionSerialization(TestBase):
    flow = None
    model_mx = None

    def setUp(self, n_levels: int = 1):
        super().setUp(n_levels=2)
        self.sklearnModel = SklearnModel('true', '1', '0.1')

        self.extension = OnnxExtension()
        config.server = self.production_server
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
        create_onnx_file(input_length, output_length, X_train, task)
        self.model_mx = onnx.load('model.onnx')
        self.flow = self.extension.model_to_flow(self.model_mx)

    def tearDown(self):
        remove_onnx_file()

    def test_is_estimator(self):
        is_estimator = self.extension.is_estimator(self.model_mx)
        is_not_estimator = self.extension.is_estimator(self.sklearnModel)

        self.assertTrue(is_estimator)
        self.assertFalse(is_not_estimator)

    def test_can_handle_model(self):
        is_onnx = self.extension.can_handle_model(self.model_mx)
        is_not_onnx = self.extension.can_handle_model(self.sklearnModel)

        self.assertTrue(is_onnx)
        self.assertFalse(is_not_onnx)

    def test_get_version_information(self):
        self.version_information = self.extension.get_version_information()

        self.assertIsInstance(self.version_information, list)
        self.assertIsNotNone(self.version_information)

    def test_create_setup_string(self):
        self.setupStringResult = self.extension.create_setup_string(self.model_mx)
        self.assertIsInstance(self.setupStringResult, str)

    def test__is_onnx_flow(self):
        self.sklearn_dummy_model = pipeline.Pipeline(
            steps=[
                ('imputer', Imputer()),
                ('estimator', tree.DecisionTreeClassifier())
            ]
        )

        self.onnx_flow_external_version = self.extension._is_onnx_flow(self.flow)

        self.sklearn_flow = SklearnExtension().model_to_flow(self.sklearn_dummy_model)
        self.sklearn_flow_external_version = self.extension._is_onnx_flow(self.sklearn_flow)

        self.assertTrue(self.onnx_flow_external_version)
        self.assertFalse(self.sklearn_flow_external_version)