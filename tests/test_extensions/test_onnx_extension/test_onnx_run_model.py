import os
import sys

import onnx
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

import openml
from openml.extensions.onnx import OnnxExtension
from openml.tasks import OpenMLRegressionTask, OpenMLClassificationTask
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestOnnxExtensionRunFunctions(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        self.extension = OnnxExtension()

    def tearDown(self):
        self._remove_mxnet_files()
        self._remove_onnx_file()

    def test_run_model_on_fold_classification(self):
        """ Function testing run_model_on_fold
        Classification task.

        :return: Nothing
        """

        # Test tasks
        task_lst = [10101, 9914, 145804, 146065, 146064]

        # Subtests, q for each task
        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                # Obtain train and test data
                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]

                # Calculate input and output shapes
                input_length = X_train.shape[1]
                output_length = len(task.class_labels)

                # Create an ONNX file from an MXNet model
                self._create_onnx_file(input_length, output_length, X_train, task)

                # Load the ONNX model from the file and remove the file
                model = onnx.load('model.onnx')
                self._remove_onnx_file()

                # Call tested functions with given model and parameters
                res = self.extension._run_model_on_fold(
                    model=model,
                    task=task,
                    fold_no=0,
                    rep_no=0,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )

                y_hat, y_hat_proba, user_defined_measures, trace = res

                # Predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsInstance(y_hat_proba, np.ndarray)
                self.assertEqual(y_hat_proba.shape, (y_test.shape[0], output_length))
                np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1),
                                                     np.ones(y_test.shape))

                # Trace comparison (Assert to None)
                self.assertIsNone(trace)

    def test_run_model_on_fold_regression(self):
        """ Function testing run_model_on_fold
        Regression task.

        :return: Nothing
        """

        task_lst = [4823, 52948, 2285, 4729, 4990]

        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                # Obtain train and test data
                X, y = task.get_X_and_y()

                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]

                # Calculate input and output shapes
                input_length = X_train.shape[1]
                output_length = 1

                # Create an ONNX file from an MXNet model
                self._create_onnx_file(input_length, output_length, X_train, task)

                # Load the ONNX model from the file and remove the file
                model = onnx.load('model.onnx')
                self._remove_onnx_file()

                # Call tested functions with given model and parameters
                res = self.extension._run_model_on_fold(
                    model=model,
                    task=task,
                    fold_no=0,
                    rep_no=0,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )

                y_hat, y_hat_proba, user_defined_measures, trace = res

                # Predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsNone(y_hat_proba)

                # Trace comparison (Assert to None)
                self.assertIsNone(trace)

    def _create_onnx_file(self, input_len, output_len, X_train, task):
        data = mx.sym.var('data')
        bnorm = mx.sym.BatchNorm(data=data)
        fc1 = mx.sym.FullyConnected(data=bnorm, num_hidden=1024)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        drop1 = mx.sym.Dropout(data=act1, p=0.1)
        fc2 = mx.sym.FullyConnected(data=drop1, num_hidden=1024)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        drop2 = mx.sym.Dropout(data=act2, p=0.2)
        fc2 = mx.sym.FullyConnected(data=drop2, num_hidden=output_len)

        if isinstance(task, OpenMLClassificationTask):
            mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
        elif isinstance(task, OpenMLRegressionTask):
            mlp = fc2

        mlp_model = mx.mod.Module(symbol=mlp, data_names=['data'], context=mx.cpu())

        data_shapes = [('data', X_train.shape)]

        mlp_model.bind(data_shapes=data_shapes)
        init = mx.init.Xavier()
        mlp_model.init_params(initializer=init)

        mlp_model.save_params('./model-0001.params')
        mlp.save('./model-symbol.json')

        onnx_mxnet.export_model(
            sym='./model-symbol.json',
            params='./model-0001.params',
            input_shape=[(1024, input_len)],
            onnx_file_path='model.onnx')

        self._remove_mxnet_files()

    def _remove_mxnet_files(self):
        if os.path.exists("model-0001.params"):
            os.remove("model-0001.params")

        if os.path.exists("model-symbol.json"):
            os.remove("model-symbol.json")

    def _remove_onnx_file(self):
        if os.path.exists("model.onnx"):
            os.remove("model.onnx")
