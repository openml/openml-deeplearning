import plotly.graph_objs as go

from visualization.tests.base import (
    VisualizationTestBase,
    ERROR_MESSAGE,
    DISPLAY_KEY
)

from visualization.utils import (
    has_error_or_is_loading,
    get_info_text_styles,
    get_loading_info,
    get_error_text,
    get_visibility_style,
    create_figure,
    extract_run_graph_data
)


class TestVisualizationUtils(VisualizationTestBase):
    def setUp(self):
        super().setUp()

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
        self.assertEqual(no_data_result, '')

        # Data is empty, so error text should be empty string
        empty_data_result = get_error_text(self.empty_data)
        self.assertEqual(empty_data_result, '')

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
        figure = create_figure(self.non_empty_graph_data, 'data')

        # Assert the figure is a dictionary, containing data and layout of the correct types
        self.assertIsInstance(figure, dict)
        self.assertIsInstance(figure['data'], go.Scatter)
        self.assertEqual(figure['data'], self.non_empty_graph_data)
        self.assertIsNotNone(figure['layout'])
        self.assertIsInstance(figure['layout'], go.Layout)

    def test_extract_run_graph_data(self):
        data = {
            'mse': [
                {'x': [1], 'y': [1], 'name': 'Name1'},
                {'x': [2], 'y': [2], 'name': 'Name2'}
            ],
            'mae': [
                {'x': [3], 'y': [3], 'name': 'Name3'}
            ]
        }

        mean_square_error_data = extract_run_graph_data(data, 'mse')
        mean_absolute_error_data = extract_run_graph_data(data, 'mae')

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
