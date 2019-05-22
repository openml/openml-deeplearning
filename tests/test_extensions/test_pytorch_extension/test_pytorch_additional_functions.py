import unittest
from distutils.version import LooseVersion
from unittest import mock

from torch import nn
import sklearn
from sklearn import pipeline
from sklearn import tree

from openml.extensions.pytorch import PytorchExtension
from openml.extensions.sklearn import SklearnExtension
from openml.setups.setup import OpenMLParameter

if LooseVersion(sklearn.__version__) < "0.20":
    from sklearn.preprocessing import Imputer
else:
    from sklearn.impute import SimpleImputer as Imputer


class SklearnModel(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, x, y):
        pass


class PytorchModel(nn.Module):

    def forward(self, *input):
        pass


class TestPytorchExtensionAdditionalFunctions(unittest.TestCase):

    def setUp(self):
        self.sklearnModel = SklearnModel('true', '1', '0.1')
        self.pytorchModel = PytorchModel()

        self.extension = PytorchExtension()

        self.pytorch_dummy_model = nn.Sequential(
            nn.LayerNorm(20),
            nn.Linear(20, 2),
            nn.Dropout(),
            nn.Softmax(dim=0)
        )

    def test_can_handle_model(self):
        is_pytorch = self.extension.can_handle_model(self.pytorchModel)
        is_not_pytorch = self.extension.can_handle_model(self.sklearnModel)

        self.assertTrue(is_pytorch)
        self.assertFalse(is_not_pytorch)

    def test_get_version_information(self):
        self.version_information = self.extension.get_version_information()

        self.assertIsInstance(self.version_information, list)
        self.assertIsNotNone(self.version_information)

    def test_create_setup_string(self):
        self.setupStringResult = self.extension.create_setup_string(self.pytorchModel)
        self.assertIsInstance(self.setupStringResult, str)

    def test_is_estimator(self):
        is_estimator = self.extension.is_estimator(self.pytorchModel)
        is_not_estimator = self.extension.is_estimator(self.sklearnModel)

        self.assertTrue(is_estimator)
        self.assertFalse(is_not_estimator)

    def test__is_pytorch_flow(self):
        self.sklearn_dummy_model = pipeline.Pipeline(
            steps=[
                ('imputer', Imputer()),
                ('estimator', tree.DecisionTreeClassifier())
            ]
        )

        self.pytorch_flow = self.extension.model_to_flow(self.pytorch_dummy_model)
        self.pytorch_flow_external_version = self.extension._is_pytorch_flow(self.pytorch_flow)

        self.sklearn_flow = SklearnExtension().model_to_flow(self.sklearn_dummy_model)
        self.sklearn_flow_external_version = self.extension._is_pytorch_flow(self.sklearn_flow)

        self.assertTrue(self.pytorch_flow_external_version)
        self.assertFalse(self.sklearn_flow_external_version)

    @mock.patch('warnings.warn')
    def test__check_dependencies(self, warnings_mock):
        dependencies = ['torch==0.1', 'torch>=99.99.99',
                        'torch>99.99.99']
        for dependency in dependencies:
            self.assertRaises(ValueError, self.extension._check_dependencies, dependency)

    def test__openml_param_name_to_torch(self):
        openml_param_dummy = OpenMLParameter(1,
                                             9763,
                                             "torch.nn.modules.container.Sequential",
                                             "torch.nn.modules.container.Sequential(("
                                             "0=torch.nn.modules.normalization.LayerNorm,"
                                             "1=torch.nn.modules.linear.Linear,"
                                             "2=torch.nn.modules.activation.ReLU,"
                                             "3=torch.nn.modules.dropout.Dropout,4=torch.nn."
                                             "modules.linear.Linear,"
                                             "5=torch.nn.modules.activation.Softmax))",
                                             "backend", "", "", "")

        self.pytorch_flow = self.extension.model_to_flow(self.pytorch_dummy_model)

        # Check if exception is thrown when OpenMLParam and Flow do not correspond
        self.assertRaises(ValueError, self.extension._openml_param_name_to_pytorch,
                          openml_param_dummy, self.pytorch_flow)
