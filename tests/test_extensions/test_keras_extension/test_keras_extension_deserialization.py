import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings
import pickle

import numpy as np
import pandas as pd
import scipy.sparse

import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

import openml
from openml.extensions.keras import KerasExtension
from openml.flows.functions import assert_flows_equal
from openml.testing import TestBase

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory)

__version__ = 0.1

class TestKerasExtensionFlowFunctions(TestBase):


    def setUp(self):
        super().setUp(n_levels=2)

        self.extension = KerasExtension()

    def test_deserialize_sequential_with_defaults(self):
        # used the 'initialize_with_defaults' flag of the deserialization
        # method to return a flow that contains default hyperparameter
        # settings.
        sequential_orig = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=1024, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(units=2, activation=keras.activations.softmax),
        ])

        # sequential_orig.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])

        # This might look like a hack, and it is, but it maintains the compilation status,
        # in contrast to clone_model, and also is faster than using get_config + load_from_config
        # since it avoids string parsing
        sequential_adjusted = pickle.loads(pickle.dumps(sequential_orig))

        # we want to confirm that sequential_adjusted and sequential_orig are the same.
        # We use the flow equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(sequential_orig),
            self.extension.model_to_flow(sequential_adjusted),
        )

        flow = self.extension.model_to_flow(sequential_adjusted)
        sequential_deserialized = self.extension.flow_to_model(flow, initialize_with_defaults=True)

        # we want to compare sequential_deserialized and sequential_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(sequential_orig),
            self.extension.model_to_flow(sequential_deserialized),
        )

    def test_deserialize_functional_with_defaults(self):
        # used the 'initialize_with_defaults' flag of the deserialization
        # method to return a flow that contains default hyperparameter
        # settings.
        main_input = Input(shape=(100,), dtype='int32', name='main_input')
        x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
        lstm_out = LSTM(32)(x)
        auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = keras.layers.concatenate([lstm_out, auxiliary_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)
        functional_orig = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

        # This might look like a hack, and it is, but it maintains the compilation status,
        # in contrast to clone_model, and also is faster than using get_config + load_from_config
        # since it avoids string parsing
        functional_adjusted = pickle.loads(pickle.dumps(functional_orig))

        # We want to confirm that functional_adjusted and functional_orig are the same.
        # We use the flow equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(functional_orig),
            self.extension.model_to_flow(functional_adjusted),
        )

        flow = self.extension.model_to_flow(functional_adjusted)
        functional_deserialized = self.extension.flow_to_model(flow, initialize_with_defaults=True)

        # we want to compare functional_deserialized and functional_orig. We use the flow
        # equals function for this
        assert_flows_equal(
            self.extension.model_to_flow(functional_orig),
            self.extension.model_to_flow(functional_deserialized),
        )

    def test_get_config(self):
        model = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=1024, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(units=2, activation=keras.activations.softmax),
        ])
        config_orig = model.get_config()
        parameters = self.extension._get_parameters(model)
        config_new = self.extension._get_config(parameters)
        self.assertEqual(config_orig, config_new)

    def test_get_parameters(self):
        model = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=1024, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.4),
            keras.layers.Dense(units=2, activation=keras.activations.softmax),
        ])
        self.extension._get_parameters(model)