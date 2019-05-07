import numpy as np
import scipy.optimize
import scipy.stats
import collections
import json
import os
import sys
import unittest
from distutils.version import LooseVersion
from collections import OrderedDict
from unittest import mock
import warnings



import openml
import keras
from keras.layers import *
from keras.models import Model
from openml.extensions.keras import KerasExtension
from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow
from openml.flows.functions import assert_flows_equal
from openml.runs.trace import OpenMLRunTrace
from openml.testing import TestBase


this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)

def test_run_model_on_fold_classification_1(self):
    task = openml.tasks.get_task(1793)

    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0)
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    model = keras.models.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=1024, activation=keras.activations.relu),
        keras.layers.Dropout(rate=0.4),
        keras.layers.Dense(units=2, activation=keras.activations.softmax),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TODO add some mocking here to actually test the innards of this function, too!
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
    self.assertEqual(y_hat_proba.shape, (y_test.shape[0], 6))
    np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1), np.ones(y_test.shape))
    # The class '4' (at index 3) is not present in the training data. We check that the
    # predicted probabilities for that class are zero!
    np.testing.assert_array_almost_equal(y_hat_proba[:, 3], np.zeros(y_test.shape))
    for i in (0, 1, 2, 4, 5):
        self.assertTrue(np.any(y_hat_proba[:, i] != np.zeros(y_test.shape)))

    # check user defined measures
    fold_evaluations = collections.defaultdict(lambda: collections.defaultdict(dict))
    for measure in user_defined_measures:
        fold_evaluations[measure][0][0] = user_defined_measures[measure]

    # trace. SGD does not produce any
    self.assertIsNone(trace)

    self._check_fold_timing_evaluations(
        fold_evaluations,
        num_repeats=1,
        num_folds=1,
        task_type=task.task_type_id,
        check_scores=False,
    )


def test_run_model_on_fold_classification_2(self):
    task = openml.tasks.get_task(1793)

    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0)
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    inputs = Input(shape=(15,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = BatchNormalization()(inputs)
    x = Dense(1024, activation=keras.activations.relu)(x)
    x = Dropout(rate=0.4)(x)
    predictions = Dense(2, activation=keras.activations.softmax)(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TODO add some mocking here to actually test the innards of this function, too!
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
    self.assertEqual(y_hat_proba.shape, (y_test.shape[0], 2))
    np.testing.assert_array_almost_equal(np.sum(y_hat_proba, axis=1), np.ones(y_test.shape))
    for i in (0, 1):
        self.assertTrue(np.any(y_hat_proba[:, i] != np.zeros(y_test.shape)))

    # check user defined measures
    fold_evaluations = collections.defaultdict(lambda: collections.defaultdict(dict))
    for measure in user_defined_measures:
        fold_evaluations[measure][0][0] = user_defined_measures[measure]

    # check that it produced and returned a trace object of the correct length
    self.assertIsInstance(trace, OpenMLRunTrace)
    self.assertEqual(len(trace.trace_iterations), 2)

    self._check_fold_timing_evaluations(
        fold_evaluations,
        num_repeats=1,
        num_folds=1,
        task_type=task.task_type_id,
        check_scores=False,
    )


def test_run_model_on_fold_regression(self):
    # There aren't any regression tasks on the test server
    openml.config.server = self.production_server
    task = openml.tasks.get_task(2999)

    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=0, fold=0, sample=0)
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    inputs = Input(shape=(10, )) # TODO run task to find shape
    x = Dense(10, activation=keras.activations.relu)(inputs)
    predictions = Dense(1, activation=keras.activations.relu)(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TODO add some mocking here to actually test the innards of this function, too!
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

    self._check_fold_timing_evaluations(
        fold_evaluations,
        num_repeats=1,
        num_folds=1,
        task_type=task.task_type_id,
        check_scores=False,
    )