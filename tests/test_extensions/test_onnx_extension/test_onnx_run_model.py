import collections
import os
import sys

import onnx
import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet

from mxnet import nd, gluon, autograd, sym

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

    def test_run_model_on_fold_classification_1(self):
        """ Function testing run_model_on_fold
        Classification task and Sequential Model.

        :return: Nothing
        """

        # Test tasks
        task_lst = [3573, 4278, 146825]

        # Subtests, q for each task
        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices] / 255
                y_train = y[train_indices]
                X_test = X[test_indices] / 255
                y_test = y[test_indices]

                print(y_test)

                input_length = X_train.shape[1]
                output_length = len(task.class_labels)
                self.create_onnx(input_length, output_length, X_train)
                model = onnx.load('model.onnx')

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

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsInstance(y_hat_proba, np.ndarray)
                self.assertEqual(y_hat_proba.shape, (y_test.shape[0], output_length))
                np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1),
                                                     np.ones(y_test.shape))

                # check user defined measures
                fold_evaluations = collections.defaultdict(lambda: collections.defaultdict(dict))
                for measure in user_defined_measures:
                    fold_evaluations[measure][0][0] = user_defined_measures[measure]

                # Trace comparison (Assert to None)
                self.assertIsNone(trace)

    def test_run_model_on_fold_regression_1(self):
        """ Function testing run_model_on_fold
        Regression task and Sequential Model.

        :return: Nothing
        """

        task_lst = [4823, 52948, 2285, 4729, 4990]

        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices] / 255
                y_train = y[train_indices]
                X_test = X[test_indices] / 255
                y_test = y[test_indices]

                input_length = X_train.shape[1]
                output_length = 1
                self.create_onnx(input_length, output_length, X_train)
                model = onnx.load('model.onnx')

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

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsNone(y_hat_proba)

                # check user defined measures
                fold_evaluations = collections.defaultdict(lambda: collections.defaultdict(dict))
                for measure in user_defined_measures:
                    fold_evaluations[measure][0][0] = user_defined_measures[measure]

                # trace. SGD does not produce any
                self.assertIsNone(trace)

    def create_onnx(self, input_len, output_len, X_train):
        # Create simple sequential model
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(1024, activation="relu"))
            net.add(gluon.nn.Dropout(0.4))
            net.add(gluon.nn.Dense(output_len, activation="softrelu"))

        # Initialize and optimize the model
        # TODO: Initializer?
        net.initialize(mx.init.Xavier(), ctx=mx.cpu())
        net.hybridize()

        # Convert training data
        input = nd.array(X_train)

        # Train the model
        with autograd.record():
            # Forward propagation must be executed at least once before export
            output = net(input)

        # Export model
        onnx_mxnet.export_model(
            sym=net(sym.var('data')),
            params={k: v._reduce() for k, v in net.collect_params().items()},
            input_shape=[(1024, input_len)],
            onnx_file_path='model.onnx')

    def remove_onnx(self):
        if os.path.exists("model.onnx"):
            os.remove("model.onnx")
