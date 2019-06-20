import json

import sklearn
import mxnet

CHILDREN_KEY = 'children'
STYLE_KEY = 'style'
VALUES_KEY = 'values'
RESPONSE_KEY = 'response'
PROPS_KEY = 'props'
OPTIONS_KEY = 'options'
VALUE_KEY = 'value'
FIGURE_KEY = 'figure'


# Mock Sklearn model
class SklearnModel(sklearn.base.BaseEstimator):
    def __init__(self, boolean, integer, floating_point_value):
        self.boolean = boolean
        self.integer = integer
        self.floating_point_value = floating_point_value

    def fit(self, x, y):
        pass


# Mock MXNet model
class MXNetModel(mxnet.gluon.nn.HybridSequential):

    def hybrid_forward(self, F, x):
        pass


def _deserialize_two_outputs_result(result_json, key_one, field_one, key_two, field_two):
    result = json.loads(result_json)

    data_one = {}
    data_two = {}

    for key, value in result[RESPONSE_KEY].items():
        if key_one in key:
            data_one = value
        if key_two in key:
            data_two = value

    return data_one[field_one], data_two[field_two]


def deserialize_info_texts_visibility_result(result_json):
    return _deserialize_two_outputs_result(
        result_json, 'loading', STYLE_KEY, 'error', STYLE_KEY)


def deserialize_text_result(result_json):
    result = json.loads(result_json)

    return result[RESPONSE_KEY][PROPS_KEY][CHILDREN_KEY]


def deserialize_id_loads_result(result_json):
    return _deserialize_two_outputs_result(
        result_json, 'id', CHILDREN_KEY, 'loads', CHILDREN_KEY)


def deserialize_loading_info_result(result_json):
    result = json.loads(result_json)

    return result[RESPONSE_KEY][PROPS_KEY][VALUES_KEY]


def deserialize_style(result_json):
    result = json.loads(result_json)

    return result[RESPONSE_KEY][PROPS_KEY][STYLE_KEY]


def deserialize_load_flow_result(result_json):
    return _deserialize_two_outputs_result(
        result_json, 'data', CHILDREN_KEY, 'error', VALUES_KEY)


def deserialize_update_flow_graph_result(result_json):
    return _deserialize_two_outputs_result(
        result_json, 'graph', CHILDREN_KEY, 'loaded', CHILDREN_KEY)


def deserialize_load_run_result(result_json):
    result = json.loads(result_json)

    data_dict = {}
    error_check_dict = {}
    dropdown_dict = {}

    for key, value in result[RESPONSE_KEY].items():
        if 'data' in key:
            data_dict = value
        if 'error' in key:
            error_check_dict = value
        if 'dropdown' in key:
            dropdown_dict = value

    return data_dict[CHILDREN_KEY], error_check_dict[VALUES_KEY], \
        dropdown_dict[OPTIONS_KEY], dropdown_dict[VALUE_KEY]


def deserialize_update_run_graph_result(result_json):
    return _deserialize_two_outputs_result(
        result_json, 'graph', FIGURE_KEY, 'loaded', CHILDREN_KEY)


def get_mxnet_model():
    model = mxnet.gluon.nn.HybridSequential()
    with model.name_scope():
        model.add(mxnet.gluon.nn.BatchNorm())
        model.add(mxnet.gluon.nn.Dense(units=1024, activation="relu"))
        model.add(mxnet.gluon.nn.Dropout(rate=0.4))
        model.add(mxnet.gluon.nn.Dense(units=2))

    return model
