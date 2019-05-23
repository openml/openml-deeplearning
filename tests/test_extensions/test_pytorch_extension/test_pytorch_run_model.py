import os
import sys

import torch
import torch.nn
import numpy as np

import openml
import openml.extensions.pytorch.layers
from openml.extensions.pytorch import PytorchExtension
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestPytorchExtensionRunFunctions(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        self.extension = PytorchExtension()

    def test_run_model_on_fold_classification(self):
        """ Function testing run_model_on_fold for a Classification task

        :return: Nothing
        """

        # Test tasks
        task_lst = [10101, 9914, 145804, 146065, 146064]

        # Subtests, q for each task
        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]

                output_length = len(task.class_labels)

                # Basic Sequential model
                model = torch.nn.Sequential(
                    torch.nn.Linear(in_features=X_train.shape[1], out_features=2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(in_features=2, out_features=output_length),
                )

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
        """ Function testing run_model_on_fold for a Regression task

        :return: Nothing
        """

        task_lst = [4823, 52948, 2285, 4729, 4990]

        for i in range(len(task_lst)):
            with self.subTest(i=i):
                task = openml.tasks.get_task(task_lst[i])

                X, y = task.get_X_and_y()
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0, fold=0, sample=0)
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]

                model = torch.nn.Sequential(
                    torch.nn.Linear(in_features=X_train.shape[1], out_features=2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(in_features=2, out_features=1),
                )
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

                # trace. SGD does not produce any
                self.assertIsNone(trace)
