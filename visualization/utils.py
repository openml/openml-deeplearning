import os
import json
import urllib3

import numpy
import pandas as pd
import plotly.graph_objs as go

import onnx
import keras
import torch
import onnxmltools
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import tensorflow as tf
import tensorflow.keras.backend as backend
from mxnet import sym
from keras import layers, models

# Import in order to register the extensions
from openml.extensions.keras import KerasExtension  # noqa: F401
from openml.extensions.pytorch import PytorchExtension  # noqa: F401
from openml.extensions.onnx import OnnxExtension  # noqa: F401
from openml.extensions.mxnet import MXNetExtension  # noqa: F401

from visualization.constants import (
    ERROR_KEY,
    DISPLAY_NONE,
    DISPLAY_VISIBLE,
    EMPTY_TEXT,
    STATIC_PATH,
    ONNX_MODEL_PATH,
    DISPLAY_KEY
)

# Disable warnings for http requests
urllib3.disable_warnings()


def has_error_or_is_loading(n_clicks, data_json, nr_loads):
    if data_json is None or nr_loads < n_clicks:  # New run is being loaded
        return True

    data = json.loads(data_json)

    return ERROR_KEY in data.keys()


def get_info_text_styles(loading_values, error_values):
    load_style = {DISPLAY_KEY: DISPLAY_NONE, 'text-align': 'center'}
    error_style = {DISPLAY_KEY: DISPLAY_NONE, 'text-align': 'center'}

    if len(loading_values) != 0:
        load_style[DISPLAY_KEY] = DISPLAY_VISIBLE

    if len(error_values) != 0:
        error_style[DISPLAY_KEY] = DISPLAY_VISIBLE

    return load_style, error_style


def get_loading_info(n_clicks, item_id, nr_loads):
    if int(nr_loads) < int(n_clicks):  # User is loading new flow or run
        if item_id is None:
            return []

        return ['load']
    else:  # Data has finished loading
        return []


def get_error_text(data_json):
    if data_json is None:
        return EMPTY_TEXT

    data = json.loads(data_json)

    if ERROR_KEY in data.keys():
        return data[ERROR_KEY]

    return EMPTY_TEXT


def get_visibility_style(n_clicks, data_json, nr_loads, curr_style):
    if has_error_or_is_loading(n_clicks, data_json, nr_loads):
        curr_style[DISPLAY_KEY] = DISPLAY_NONE
    else:
        curr_style[DISPLAY_KEY] = DISPLAY_VISIBLE

    return curr_style


def create_figure(data, y_label):
    return {
        'data': data,
        'layout': go.Layout(
            xaxis={'title': 'Iterations'},
            yaxis={'title': y_label},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            # legend={'x': 0, 'y': 1},
            # hovermode='closest'
        )
    }


def extract_run_graph_data(run_data, key):
    data = []

    for item in run_data[key]:
        data.append(
            go.Scatter(
                x=item['x'],
                y=item['y'],
                mode='lines',
                name=item['name']
            )
        )

    return data


def get_training_data(url, run_id):
    if '.csv' not in url:
        return None

    # Create a request to the url
    http = urllib3.PoolManager()
    request = http.request('GET', url, preload_content=False)

    # Compute the path to the file
    csv_file_path = os.path.join(STATIC_PATH, 'training_{}.csv'.format(run_id))

    # Read the data from the URL and store it in a file
    with open(csv_file_path, 'wb') as out:
        while True:
            data = request.read()
            if not data:
                break
            out.write(data)

    # Release the connection
    request.release_conn()

    # Read the data from the file
    df = pd.read_csv(csv_file_path)

    # Remove the leftover file
    os.remove(csv_file_path)

    return df


def get_onnx_model(model):
    if isinstance(model, keras.models.Model):
        # Create a session to avoid problems with names
        session = tf.Session()
        backend.set_session(session)

        with session.as_default():
            with session.graph.as_default():
                # If the model is sequential, it must be executed once and converted
                # to a function one prior to exporting the ONNX model
                if isinstance(model, keras.models.Sequential):
                    model.compile('Adam')

                    # Generate a random input just to execute the model once
                    dummy_input = numpy.random.rand(2, 2)
                    model.predict(dummy_input)

                    # Find the input and output layers to construct the functional model
                    input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
                    prev_layer = input_layer
                    for layer in model.layers:
                        prev_layer = layer(prev_layer)

                    # Create a functional model equivalent to the sequential model
                    model = models.Model([input_layer], [prev_layer])

                # Export the functional keras model
                onnx_model = onnxmltools.convert_keras(model, target_opset=7)
    elif isinstance(model, torch.nn.Module):
        input_shape_found = False

        # Try to find the input shape and export the model
        for i in range(1, 5000):
            try:
                dummy_input = torch.randn(i, i, dtype=torch.float)
                torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH)
                input_shape_found = True
            except RuntimeError:
                pass

            # There was no error, so the input shape has been correctly guessed
            # and the ONNX model was exported so we can stop iterating
            if input_shape_found:
                break

        # If the input shape could not be guessed, return None
        # and an error message will be displayed to the user
        if not input_shape_found:
            return None

        # Load the exported ONNX model file and remove the left-over file
        onnx_model = onnx.load_model(ONNX_MODEL_PATH)
        os.remove(ONNX_MODEL_PATH)
    elif isinstance(model, mx.gluon.nn.HybridBlock):
        # Initialize the MXNet model and create some dummy input
        model.collect_params().initialize(mx.init.Normal())
        dummy_input = numpy.random.rand(2, 2)

        # Propagate the input forward so the model can be fully initialized
        with mx.autograd.record():
            model(mx.nd.array(dummy_input))

        # Once initialized, export the ONNX equivalent of the model
        onnx_mxnet.export_model(
            sym=model(sym.var('data')),
            params={k: v._reduce() for k, v in model.collect_params().items()},
            input_shape=[(64, 2)],
            onnx_file_path=ONNX_MODEL_PATH)

        # Load the exported ONNX model file and remove the left-over file
        onnx_model = onnx.load_model(ONNX_MODEL_PATH)
        os.remove(ONNX_MODEL_PATH)
    elif isinstance(model, onnx.ModelProto):
        # The model is already an ONNX one
        onnx_model = model
    else:
        # The model was not produced by Keras, PyTorch, MXNet or ONNX and cannot be visualized
        # This point should not be reachable, as retrieval of the flow reinitializes the model
        # and that ensures if can be handled by MXNetExtension, OnnxExtension, PytorchExtension
        # or KerasExtension and therefore the model was produced by one of the libraries.
        return None

    return onnx_model


def simplyfy_pydot_graph(pydot_graph):
    for k, v in pydot_graph.obj_dict['nodes'].items():
        if "(op" in k:
            orig_name = v[0]['name']
            v[0]['name'] = orig_name.rsplit('\\n')[0] + '\"'
    for k, v in pydot_graph.obj_dict['edges'].items():
        edge = v[0]
        lst = []
        for idx, rv in enumerate(edge['points']):
            if "(op" in rv:
                lst.append(rv.rsplit('\\n')[0] + '\"')
            else:
                lst.append(rv)
        edge['points'] = lst

    return pydot_graph


def add_lists_element_wise(lst1, lst2):
    first = lst1.copy()
    second = lst2.copy()

    if len(first) > len(second):
        # Swap the lists
        first, second = second, first

    for idx, val in enumerate(first):
        second[idx] += val

    return second
