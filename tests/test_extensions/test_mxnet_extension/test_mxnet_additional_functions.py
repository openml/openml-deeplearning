import unittest
from distutils.version import LooseVersion
from unittest import mock

from mxnet.gluon import nn
import sklearn
from sklearn import pipeline
from sklearn import tree

from openml.extensions.mxnet import MXNetExtension
from openml.extensions.sklearn import SklearnExtension

if LooseVersion(sklearn.__version__) < "0.20":
    from sklearn.preprocessing import Imputer
else:
    from sklearn.impute import SimpleImputer as Imputer


# Mock Sklearn model
class SklearnModel(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, x, y):
        pass


# Mock MXNet model
class MXNetModel(nn.HybridSequential):

    def hybrid_forward(self, F, x):
        pass


class TestMXNetExtensionAdditionalFunctions(unittest.TestCase):

    def setUp(self):
        self.sklearnModel = SklearnModel('true', '1', '0.1')
        self.mxnetModel = MXNetModel()

        self.extension = MXNetExtension()

        self.mxnet_dummy_model = nn.HybridSequential()

        with self.mxnet_dummy_model.name_scope():
            self.mxnet_dummy_model.add(
                nn.Dense(1024, activation="relu"),
                nn.Dense(2)
            )

        self.mxnet_dummy_model.hybridize()

    def test_can_handle_model(self):
        is_mxnet = self.extension.can_handle_model(self.mxnetModel)
        is_not_mxnet = self.extension.can_handle_model(self.sklearnModel)

        self.assertTrue(is_mxnet)
        self.assertFalse(is_not_mxnet)

    def test_get_version_information(self):
        self.version_information = self.extension.get_version_information()

        self.assertIsInstance(self.version_information, list)
        self.assertIsNotNone(self.version_information)

    def test_create_setup_string(self):
        self.setupStringResult = self.extension.create_setup_string(self.mxnetModel)
        self.assertIsInstance(self.setupStringResult, str)

    def test_is_estimator(self):
        is_estimator = self.extension.is_estimator(self.mxnetModel)
        is_not_estimator = self.extension.is_estimator(self.sklearnModel)

        self.assertTrue(is_estimator)
        self.assertFalse(is_not_estimator)

    def test__is_mxnet_flow(self):
        self.sklearn_dummy_model = pipeline.Pipeline(
            steps=[
                ('imputer', Imputer()),
                ('estimator', tree.DecisionTreeClassifier())
            ]
        )

        # Convert MXNet dummy model to flow
        self.mxnet_flow = self.extension.model_to_flow(self.mxnet_dummy_model)
        self.mxnet_flow_bool = self.extension._is_mxnet_flow(self.mxnet_flow)

        # Convert sklearn dummy model to flow
        self.sklearn_flow = SklearnExtension().model_to_flow(self.sklearn_dummy_model)
        self.sklearn_flow_bool = self.extension._is_mxnet_flow(self.sklearn_flow)

        # Check whether the MXNet flow is correctly recognized
        self.assertTrue(self.mxnet_flow_bool)

        # Test that the Sklearn flow is not an MXNet flow
        self.assertFalse(self.sklearn_flow_bool)

    @mock.patch('warnings.warn')
    def test__check_dependencies(self, warnings_mock):
        dependencies = ['mxnet==0.1', 'mxnet>=99.99.99',
                        'mxnet>99.99.99']
        for dependency in dependencies:
            self.assertRaises(ValueError, self.extension._check_dependencies, dependency)
