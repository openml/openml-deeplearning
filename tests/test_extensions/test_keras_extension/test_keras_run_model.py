import os
import sys

import keras
import numpy as np
from keras.layers import Input, BatchNormalization, Dense, Dropout
from keras.models import Model

import openml
from openml.extensions.keras import KerasExtension
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)


class TestKerasExtensionRunFunctions(TestBase):

    def setUp(self):
        super().setUp(n_levels=2)
        # Test server has out of date data sets, so production server is used
        openml.config.server = self.production_server

        self.extension = KerasExtension()

    def test_run_model_on_fold_classification_1(self):
        """ Function testing run_model_on_fold
        Classification task and Sequential Model.

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
                model = keras.models.Sequential([
                    keras.layers.BatchNormalization(),
                    keras.layers.Dense(units=64, activation=keras.activations.relu),
                    keras.layers.Dropout(rate=0.4),
                    keras.layers.Dense(units=output_length, activation=keras.activations.softmax),
                ])

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

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

                y_hat, y_hat_proba, user_defined_measures, trace, addinfo = res

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsInstance(y_hat_proba, np.ndarray)
                self.assertEqual(y_hat_proba.shape, (y_test.shape[0], output_length))
                np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1),
                                                     np.ones(y_test.shape))

                # Trace comparison (Assert to None)
                self.assertIsNone(trace)

    def test_run_model_on_fold_classification_2(self):
        """ Function testing run_model_on_fold
        Classification task and Functional Model.

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

                # get number of columns in training data
                n_cols = X_train.shape[1]

                inputs = Input(shape=(n_cols,))
                output_length = len(task.class_labels)

                # Basic Functional Model
                # a layer instance is callable on a tensor, and returns a tensor
                x = BatchNormalization()(inputs)
                x = Dense(64, activation=keras.activations.relu)(x)
                x = Dropout(rate=0.4)(x)
                predictions = Dense(output_length, activation=keras.activations.softmax)(x)

                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                res = self.extension._run_model_on_fold(
                    model=model,
                    task=task,
                    fold_no=0,
                    rep_no=0,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )

                y_hat, y_hat_proba, user_defined_measures, trace, addinfo = res

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsInstance(y_hat_proba, np.ndarray)
                self.assertEqual(y_hat_proba.shape, (y_test.shape[0], output_length))

                # trace should assert to None
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
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]

                model = keras.models.Sequential([
                    keras.layers.BatchNormalization(),
                    keras.layers.Dense(units=64, activation=keras.activations.relu),
                    keras.layers.Dropout(rate=0.4),
                    keras.layers.Dense(units=1, activation=keras.activations.softmax),
                ])
                model.compile(optimizer='adam',
                              loss='mean_squared_error',
                              metrics=['accuracy'])

                res = self.extension._run_model_on_fold(
                    model=model,
                    task=task,
                    fold_no=0,
                    rep_no=0,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )

                y_hat, y_hat_proba, user_defined_measures, trace, addinfo = res

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsNone(y_hat_proba)

                # trace. SGD does not produce any
                self.assertIsNone(trace)

    def test_run_model_on_fold_regression_2(self):
        """ Function testing run_model_on_fold
        Regression task and Functional Model.

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

                # get number of columns in training data
                n_cols = X_train.shape[1]

                inputs = Input(shape=(n_cols,))

                # Basic Functional Model
                x = BatchNormalization()(inputs)
                x = Dense(64, activation=keras.activations.relu)(x)
                x = Dropout(rate=0.4)(x)
                predictions = Dense(1, activation=keras.activations.softmax)(x)

                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam', loss='mean_squared_error')

                res = self.extension._run_model_on_fold(
                    model=model,
                    task=task,
                    fold_no=0,
                    rep_no=0,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )

                y_hat, y_hat_proba, user_defined_measures, trace, addinfo = res

                # predictions
                self.assertIsInstance(y_hat, np.ndarray)
                self.assertEqual(y_hat.shape, y_test.shape)
                self.assertIsNone(y_hat_proba)

                # Trace should assert to None
                self.assertIsNone(trace)
