import os

from openml.extensions.pytorch.layers import Functional
import onnx
import keras
import torch
import pandas as pd
import plotly.graph_objs as go

from visualization.tests.base import (
    VisualizationTestBase,
    ERROR_MESSAGE,
    DISPLAY_KEY
)

from visualization.tests.utils import (
    SklearnModel,
    get_mxnet_model
)

from visualization.utils import (
    has_error_or_is_loading,
    get_info_text_styles,
    get_loading_info,
    get_error_text,
    get_visibility_style,
    create_figure,
    extract_run_graph_data,
    add_lists_element_wise,
    get_training_data,
    get_onnx_model
)

from visualization.constants import (
    STATIC_PATH,
    ONNX_MODEL_PATH,
    EMPTY_TEXT,
    MEAN_SQUARE_ERROR_KEY,
    MEAN_ABSOLUTE_ERROR_KEY
)

DATA_KEY = 'data'


class TestVisualizationUtils(VisualizationTestBase):
    def setUp(self):
        super().setUp()

        self.sklearn_model = SklearnModel('true', '1', '0.1')
        self.keras_model = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=20, activation=keras.activations.relu)
        ])
        self.pytorch_model = torch.nn.Sequential(
            Functional(function=torch.Tensor.reshape, shape=(-1, 1, 28, 28)),
            torch.nn.BatchNorm2d(num_features=1)
        )
        self.mxnet_model = get_mxnet_model()

    def test_has_error_or_is_loading(self):
        # Nothing is being displayed, so assume loading
        none_data_same_clicks_and_loads_result = has_error_or_is_loading(0, self.none_data, 0)
        self.assertTrue(none_data_same_clicks_and_loads_result)

        # The load button has been clicked, but data has not been loaded, so should be loading
        none_data_different_clicks_and_loads_1_result = \
            has_error_or_is_loading(1, self.none_data, 0)
        self.assertTrue(none_data_different_clicks_and_loads_1_result)

        # Impossible case since number of loads should always be less than the number of clicks
        # on the load button, but default behavior should return true
        none_data_different_clicks_and_loads_2_result = \
            has_error_or_is_loading(0, self.none_data, 1)
        self.assertTrue(none_data_different_clicks_and_loads_2_result)

        # The load button has been clicked, and the data is not fully loaded, so should be true
        no_error_data_different_clicks_and_loads_result = \
            has_error_or_is_loading(1, self.empty_data, 0)
        self.assertTrue(no_error_data_different_clicks_and_loads_result)

        # There is data without errors and the number of clicks and loads is the same so it's
        # neither loading, nor has error
        no_error_data_same_clicks_and_loads_result = has_error_or_is_loading(0, self.empty_data, 0)
        self.assertFalse(no_error_data_same_clicks_and_loads_result)

        # Both has error in the current data, and new data is being loaded
        # (due to difference in n_clicks and nr_loads)
        error_data_different_clicks_and_loads_result = \
            has_error_or_is_loading(1, self.error_data, 0)
        self.assertTrue(error_data_different_clicks_and_loads_result)

        # The data is loaded, but has an error
        error_data_same_clicks_and_loads_result = has_error_or_is_loading(0, self.error_data, 0)
        self.assertTrue(error_data_same_clicks_and_loads_result)

    def test_get_info_text_styles(self):
        # There are no errors and nothing is loading so texts should be invisible
        load_style, error_style = get_info_text_styles(self.empty_loading, self.empty_error)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are no errors, but data is loading, so loading text should be visible
        load_style, error_style = get_info_text_styles(self.non_empty_loading, self.empty_error)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are errors and no data is loading, so error text should be visible
        # and loading text should be hidden
        load_style, error_style = get_info_text_styles(self.empty_loading, self.non_empty_error)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

        # There are errors and data is loading, so both texts should be visible
        load_style, error_style = get_info_text_styles(self.non_empty_loading, self.non_empty_error)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

    def test_get_loading_info(self):
        # There is no id, so data is not being loaded
        more_clicks_than_loaded_but_no_id_result = get_loading_info(1, None, 0)
        self.assertEqual(more_clicks_than_loaded_but_no_id_result, self.empty_loading)

        # Load button has been clicked and the data is still not loaded
        # (but there is an id, which indicates it is being loaded)
        more_clicks_than_loaded_and_id_result = get_loading_info(1, 1, 0)
        self.assertEqual(more_clicks_than_loaded_and_id_result, self.non_empty_loading)

    def test_get_error_text(self):
        # There is no data, so error message should be empty string
        no_data_result = get_error_text(self.none_data)
        self.assertEqual(no_data_result, EMPTY_TEXT)

        # Data is empty, so error text should be empty string
        empty_data_result = get_error_text(self.empty_data)
        self.assertEqual(empty_data_result, EMPTY_TEXT)

        # Data has error, so the text should be equal to the error message in the data
        error_data_result = get_error_text(self.error_data)
        self.assertEqual(error_data_result, ERROR_MESSAGE)

    def test_get_visibility_style(self):
        # Nothing is being displayed, so text should be hidden
        none_data_same_clicks_and_loads_currently_visible_result = \
            get_visibility_style(0, self.none_data, 0, self.visible_style)
        none_data_same_clicks_and_loads_currently_hidden_result = \
            get_visibility_style(0, self.none_data, 0, self.visible_style)
        self.assertEqual(none_data_same_clicks_and_loads_currently_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(none_data_same_clicks_and_loads_currently_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

        # The load button has been clicked, but data has not been loaded, so text should be hidden
        none_data_different_clicks_and_loads_1_curr_visible_result = \
            get_visibility_style(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_1_curr_hidden_result = \
            get_visibility_style(1, self.none_data, 0, self.hidden_style)
        self.assertEqual(none_data_different_clicks_and_loads_1_curr_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(none_data_different_clicks_and_loads_1_curr_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

        # Impossible case since number of loads should always be less than the number of clicks
        # on the load button, but by default text should be hidden
        none_data_different_clicks_and_loads_2_curr_visible_result = \
            get_visibility_style(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_2_curr_hidden_result = \
            get_visibility_style(1, self.none_data, 0, self.hidden_style)
        self.assertEqual(none_data_different_clicks_and_loads_2_curr_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(none_data_different_clicks_and_loads_2_curr_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

        # The load button has been clicked, but the data is not fully loaded,
        # so text should be hidden
        no_error_data_different_clicks_and_loads_curr_visible_result = \
            get_visibility_style(1, self.empty_data, 0, self.visible_style)
        no_error_data_different_clicks_and_loads_curr_hidden_result = \
            get_visibility_style(1, self.empty_data, 0, self.visible_style)
        self.assertEqual(no_error_data_different_clicks_and_loads_curr_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(no_error_data_different_clicks_and_loads_curr_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

        # There is data without errors and the number of clicks and loads is the same so it's
        # neither loading, nor has error - text should be visible
        no_error_data_same_clicks_and_loads_curr_visible_result = \
            get_visibility_style(0, self.empty_data, 0, self.visible_style)
        no_error_data_same_clicks_and_loads_curr_hidden_result = \
            get_visibility_style(0, self.empty_data, 0, self.hidden_style)
        self.assertEqual(no_error_data_same_clicks_and_loads_curr_visible_result[DISPLAY_KEY],
                         self.display_visible)
        self.assertEqual(no_error_data_same_clicks_and_loads_curr_hidden_result[DISPLAY_KEY],
                         self.display_visible)

        # Both has error in the current data, and new data is being loaded
        # (due to difference in n_clicks and nr_loads) - text should be hidden
        error_data_different_clicks_and_loads_curr_visible_result = \
            get_visibility_style(1, self.error_data, 0, self.visible_style)
        error_data_different_clicks_and_loads_curr_hidden_result = \
            get_visibility_style(1, self.error_data, 0, self.hidden_style)
        self.assertEqual(error_data_different_clicks_and_loads_curr_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(error_data_different_clicks_and_loads_curr_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

        # The data is loaded, but has an error - text should be hidden
        error_data_same_clicks_and_loads_curr_visible_result = \
            get_visibility_style(0, self.error_data, 0, self.visible_style)
        error_data_same_clicks_and_loads_curr_hidden_result = \
            get_visibility_style(0, self.error_data, 0, self.visible_style)
        self.assertEqual(error_data_same_clicks_and_loads_curr_visible_result[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(error_data_same_clicks_and_loads_curr_hidden_result[DISPLAY_KEY],
                         self.display_hidden)

    def test_create_figure(self):
        figure = create_figure(self.non_empty_graph_data, DATA_KEY)

        # Assert the figure is a dictionary, containing data and layout of the correct types
        self.assertIsInstance(figure, dict)
        self.assertIsInstance(figure[DATA_KEY], go.Scatter)
        self.assertEqual(figure[DATA_KEY], self.non_empty_graph_data)
        self.assertIsNotNone(figure['layout'])
        self.assertIsInstance(figure['layout'], go.Layout)

    def test_extract_run_graph_data(self):
        data = {
            MEAN_SQUARE_ERROR_KEY: [
                {'x': [1], 'y': [1], 'name': 'Name1'},
                {'x': [2], 'y': [2], 'name': 'Name2'}
            ],
            MEAN_ABSOLUTE_ERROR_KEY: [
                {'x': [3], 'y': [3], 'name': 'Name3'}
            ]
        }

        mean_square_error_data = extract_run_graph_data(data, MEAN_SQUARE_ERROR_KEY)
        mean_absolute_error_data = extract_run_graph_data(data, MEAN_ABSOLUTE_ERROR_KEY)

        # Assert the resulted values are lists
        self.assertIsInstance(mean_square_error_data, list)
        self.assertIsInstance(mean_absolute_error_data, list)

        # Assert only the data from the appropriate key is extracted
        self.assertEqual(len(mean_square_error_data), 2)
        self.assertEqual(len(mean_absolute_error_data), 1)

        # Assert the lists are filled will instances of go.Scatter
        for item in mean_square_error_data:
            self.assertIsInstance(item, go.Scatter)
        for item in mean_absolute_error_data:
            self.assertIsInstance(item, go.Scatter)

    def test_add_lists_element_wise(self):
        # Adding two empty lists should result in an empty list
        result = add_lists_element_wise([], [])
        self.assertListEqual(result, [])

        # Adding an empty first list to a non-empty second list should work correctly
        result = add_lists_element_wise([], [1, 2, 3])
        self.assertListEqual(result, [1, 2, 3])

        # Adding a non-empty first list to an empty second list should work correctly
        result = add_lists_element_wise([1, 2, 3], [])
        self.assertListEqual(result, [1, 2, 3])

        # Adding two non-empty lists where first one is longer should work correctly
        result = add_lists_element_wise([1, 2, 3], [4, 5])
        self.assertListEqual(result, [5, 7, 3])

        # Adding two non-empty lists where second one is longer should work correctly
        result = add_lists_element_wise([1, 2], [3, 4, 5])
        self.assertListEqual(result, [4, 6, 5])

        # Adding two lists of equal length should work correctly
        result = add_lists_element_wise([1, 2, 3], [4, 5, 6])
        self.assertListEqual(result, [5, 7, 9])

        # Adding the two lists should not modify them
        list1_orig = [1, 2, 3]
        list2_orig = [3, 4, 5]
        list1_copy = list1_orig.copy()
        list2_copy = list2_orig.copy()

        add_lists_element_wise(list1_copy, list2_copy)
        self.assertListEqual(list1_orig, list1_copy)
        self.assertListEqual(list2_orig, list2_copy)

    def test_get_training_data(self):
        # Retriving a url which is not to a training data should result in none
        result = get_training_data('https://www.google.com', self.run_id)
        self.assertIsNone(result)

        # Retrieving a url which is of a training data, should
        # return the data as a pandas DataFrame
        result = get_training_data('https://www.openml.org/data/download/21378680/training.csv',
                                   self.run_id)

        csv_file_path = os.path.join(STATIC_PATH, 'training_{}.csv'.format(self.run_id))
        csv_file_exists = os.path.exists(csv_file_path)

        # Ensure the file is deleted after retrieval
        self.assertFalse(csv_file_exists)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_onnx_model(self):
        # After every call it is asserted that there are no left-over
        # ONNX files in the static directory

        # Passing a keras model should return an onnx model
        result = get_onnx_model(self.keras_model)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, onnx.ModelProto)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))

        # Passing a pytorch model should return an onnx model
        result = get_onnx_model(self.pytorch_model)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, onnx.ModelProto)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))

        # Passing a mxnet model should return an onnx model
        result = get_onnx_model(self.mxnet_model)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, onnx.ModelProto)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))

        # Passing an onnx model should return the passed model
        result = get_onnx_model(self.onnx_model)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, onnx.ModelProto)
        self.assertEqual(result, self.onnx_model)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))

        # Passing an sklearn model should return None
        result = get_onnx_model(self.sklearn_model)
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))

        # Passing None should return None
        result = get_onnx_model(None)
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(ONNX_MODEL_PATH))
