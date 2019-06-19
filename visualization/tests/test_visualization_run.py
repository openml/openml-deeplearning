import json

from visualization.tests.base import (
    VisualizationTestBase,
    MEAN_SQUARE_ERROR,
    MEAN_ABSOLUTE_ERROR,
    ERROR_MESSAGE,
    DISPLAY_KEY
)

from visualization.constants import (
    LOADING_TEXT_RUN_INFO,
    RUN_GRAPH_TEXT_TEMPLATE,
    METRIC_TO_LABEL,
    EMPTY_SELECTION,
    ERROR_KEY,
    EMPTY_LOADED,
    RUN_ID_KEY,
    LOSS_KEY
)

from visualization.visualizer import (
    update_run_info_texts_visibility,
    update_run_graph_text,
    init_run_loading,
    update_run_loading_info,
    update_run_error_text,
    update_run_graph_visibility,
    load_run,
    update_run_graph
)

from visualization.tests.utils import (
    deserialize_style,
    deserialize_text_result,
    deserialize_loading_info_result,
    deserialize_id_loads_result,
    deserialize_info_texts_visibility_result,
    deserialize_load_run_result,
    deserialize_update_run_graph_result
)


DATA_KEY = 'data'


class TestVisualizationRun(VisualizationTestBase):
    def setUp(self):
        super().setUp()

        self.training_data_run = 10228688
        self.clustering_run = 7951657
        self.no_training_data_run = 10228348

        self.loss_data = {
            RUN_ID_KEY: self.run_id,
            LOSS_KEY: [{
                'x': [1, 2, 3],
                'y': [1, 2, 3],
                'name': ''
            }]
        }

        super().setUp()

    def test_update_run_info_texts_visibility(self):
        # There are no errors and nothing is loading so texts should be invisible
        result_json = update_run_info_texts_visibility(self.empty_loading, self.empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are no errors, but run data is loading, so loading text should be visible
        result_json = update_run_info_texts_visibility(self.non_empty_loading, self.empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are errors and no run data is loading, so error text should be visible
        # and loading text should be hidden
        result_json = update_run_info_texts_visibility(self.empty_loading, self.non_empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

        # There are errors and run data is loading, so both texts should be visible
        result_json = update_run_info_texts_visibility(self.non_empty_loading, self.non_empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

    def test_update_run_graph_text(self):
        # There is no data, so the text should be empty
        none_data_same_metric_and_loaded_result = \
            update_run_graph_text(MEAN_SQUARE_ERROR, MEAN_SQUARE_ERROR, self.none_data)
        result = deserialize_text_result(none_data_same_metric_and_loaded_result)
        self.assertEqual(result, '')

        # There is no selected metric, so the text should be empty
        data_no_metric_result = update_run_graph_text('', '', self.simple_data)
        result = deserialize_text_result(data_no_metric_result)
        self.assertEqual(result, '')

        # Metric has changed so new figure is being loaded and the loading text should be displayed
        data_metric_different_from_loaded_result = \
            update_run_graph_text(MEAN_SQUARE_ERROR, MEAN_ABSOLUTE_ERROR, self.simple_data)
        result = deserialize_text_result(data_metric_different_from_loaded_result)
        self.assertEqual(result, LOADING_TEXT_RUN_INFO)

        # The run is loaded so the text for a loaded run should be displayed
        data_metric_same_as_loaded_result = \
            update_run_graph_text(MEAN_SQUARE_ERROR, MEAN_SQUARE_ERROR, self.simple_data)
        result = deserialize_text_result(data_metric_same_as_loaded_result)
        self.assertEqual(
            result, RUN_GRAPH_TEXT_TEMPLATE.format(METRIC_TO_LABEL[MEAN_SQUARE_ERROR], self.run_id))

    def test_init_run_loading(self):
        # The function is used to pass data along, so the result should be same as input params

        # Tests with no run id provided
        for n_clicks in range(10):
            no_run_id_result = init_run_loading(n_clicks, None)
            id_result, loads_result = deserialize_id_loads_result(no_run_id_result)

            self.assertEqual(id_result, None)
            self.assertEqual(loads_result, n_clicks)

        # Test with run id provided
        for n_clicks in range(10):
            run_id_result = init_run_loading(n_clicks, self.run_id)
            id_result, loads_result = deserialize_id_loads_result(run_id_result)

            self.assertEqual(id_result, self.run_id)
            self.assertEqual(loads_result, n_clicks)

    def test_update_run_loading_info(self):
        # There is no id, so run data is not being loaded
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_but_no_id_result = \
            update_run_loading_info(1, self.simple_data, None, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_but_no_id_result)
        self.assertEqual(result, self.empty_loading)

        # Load button has been clicked and the run data is still not loaded
        # (but there is a run id, which indicates it is being loaded)
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_and_id_result = update_run_loading_info(1, self.simple_data, 1, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_and_id_result)
        self.assertEqual(result, self.non_empty_loading)

    def test_load_run(self):
        # Passing None for run id should return None data, empty error check
        # empty dropdown options and empty dropdown value
        result_json = load_run(None)
        data_json, error_check, dropdown_options, dropdown_value = \
            deserialize_load_run_result(result_json)
        self.assertIsInstance(error_check, list)
        self.assertEqual(len(error_check), 0)

        # Check that dropdown options are empty and the selection is empty
        self.assertIsInstance(dropdown_options, list)
        self.assertIsInstance(dropdown_value, str)
        self.assertEqual(len(dropdown_options), 0)
        self.assertEqual(dropdown_value, EMPTY_SELECTION)

        self.assertIsNone(data_json)

        # Passing a non-existent run id should return an error
        result_json = load_run(0)
        data_json, error_check, dropdown_options, dropdown_value = \
            deserialize_load_run_result(result_json)
        self.assertIsInstance(error_check, list)
        self.assertEqual(len(error_check), 1)

        # Check that dropdown options are empty and the selection is empty
        self.assertIsInstance(dropdown_options, list)
        self.assertIsInstance(dropdown_value, str)
        self.assertEqual(len(dropdown_options), 0)
        self.assertEqual(dropdown_value, EMPTY_SELECTION)

        self.assertIsNotNone(data_json)
        data = json.loads(data_json)
        self.assertTrue(ERROR_KEY in data.keys())

        # Passing the id of a run without associated with a task that is neither
        # classification nor regression should return an error
        result_json = load_run(self.clustering_run)
        data_json, error_check, dropdown_options, dropdown_value = \
            deserialize_load_run_result(result_json)
        self.assertIsInstance(error_check, list)
        self.assertEqual(len(error_check), 1)

        # Check that dropdown options are empty and the selection is empty
        self.assertIsInstance(dropdown_options, list)
        self.assertIsInstance(dropdown_value, str)
        self.assertEqual(len(dropdown_options), 0)
        self.assertEqual(dropdown_value, EMPTY_SELECTION)

        self.assertIsNotNone(data_json)
        data = json.loads(data_json)
        self.assertTrue(ERROR_KEY in data.keys())

        # Passing the id of a run which does not have training data should return an error
        result_json = load_run(self.no_training_data_run)
        data_json, error_check, dropdown_options, dropdown_value = \
            deserialize_load_run_result(result_json)
        self.assertIsInstance(error_check, list)
        self.assertEqual(len(error_check), 1)

        # Check that dropdown options are empty and the selection is empty
        self.assertIsInstance(dropdown_options, list)
        self.assertIsInstance(dropdown_value, str)
        self.assertEqual(len(dropdown_options), 0)
        self.assertEqual(dropdown_value, EMPTY_SELECTION)

        self.assertIsNotNone(data_json)
        data = json.loads(data_json)
        self.assertTrue(ERROR_KEY in data.keys())
        self.assertTrue('training' in data[ERROR_KEY])

        # Passing the id of a run which has training data should return no error,
        # valid dropdown options and valid dropdown selection
        result_json = load_run(self.training_data_run)
        data_json, error_check, dropdown_options, dropdown_value = \
            deserialize_load_run_result(result_json)
        self.assertIsInstance(error_check, list)
        self.assertEqual(len(error_check), 0)

        # Check that dropdown options are empty and the selection is empty
        self.assertIsInstance(dropdown_options, list)
        self.assertIsInstance(dropdown_value, str)
        self.assertGreater(len(dropdown_options), 0)
        self.assertGreater(len(dropdown_value), 0)

        self.assertIsNotNone(data_json)
        data = json.loads(data_json)
        self.assertFalse(ERROR_KEY in data.keys())

        # Make sure all options are present in the data
        for option in dropdown_options:
            self.assertTrue(option['value'] in data.keys())

        # Make sure the selected dropdown value is present in the data
        self.assertTrue(dropdown_value in data.keys())

    def test_update_run_error_text(self):
        # There is no run data, so error message should be empty string
        no_data_result = update_run_error_text(self.none_data)
        result = deserialize_text_result(no_data_result)
        self.assertEqual(result, '')

        # Run data is empty, so error text should be empty string
        empty_data_result = update_run_error_text(self.empty_data)
        result = deserialize_text_result(empty_data_result)
        self.assertEqual(result, '')

        # Run data has error, so the text should be equal to the error message in the data
        error_data_result = update_run_error_text(self.error_data)
        result = deserialize_text_result(error_data_result)
        self.assertEqual(result, ERROR_MESSAGE)

    def test_update_run_graph_visibility(self):
        # Nothing is being displayed, so text should be hidden
        none_data_same_clicks_and_loads_currently_visible_result = \
            update_run_graph_visibility(0, self.none_data, 0, self.visible_style)
        none_data_same_clicks_and_loads_currently_hidden_result = \
            update_run_graph_visibility(0, self.none_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(none_data_same_clicks_and_loads_currently_visible_result)
        result_hidden = \
            deserialize_style(none_data_same_clicks_and_loads_currently_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # The load button has been clicked, but run data has
        # not been loaded, so text should be hidden
        none_data_different_clicks_and_loads_1_curr_visible_result = \
            update_run_graph_visibility(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_1_curr_hidden_result = \
            update_run_graph_visibility(1, self.none_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(none_data_different_clicks_and_loads_1_curr_visible_result)
        result_hidden = \
            deserialize_style(none_data_different_clicks_and_loads_1_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # Impossible case since number of loads should always be less than the number of clicks
        # on the load button, but by default text should be hidden
        none_data_different_clicks_and_loads_2_curr_visible_result = \
            update_run_graph_visibility(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_2_curr_hidden_result = \
            update_run_graph_visibility(1, self.none_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(none_data_different_clicks_and_loads_2_curr_visible_result)
        result_hidden = \
            deserialize_style(none_data_different_clicks_and_loads_2_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # The load button has been clicked, but the data is not fully loaded,
        # so text should be hidden
        no_error_data_different_clicks_and_loads_curr_visible_result = \
            update_run_graph_visibility(1, self.empty_data, 0, self.visible_style)
        no_error_data_different_clicks_and_loads_curr_hidden_result = \
            update_run_graph_visibility(1, self.empty_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(no_error_data_different_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(no_error_data_different_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # There is data without errors and the number of clicks and loads is the same so it's
        # neither loading, nor has error - text should be visible
        no_error_data_same_clicks_and_loads_curr_visible_result = \
            update_run_graph_visibility(0, self.empty_data, 0, self.visible_style)
        no_error_data_same_clicks_and_loads_curr_hidden_result = \
            update_run_graph_visibility(0, self.empty_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(no_error_data_same_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(no_error_data_same_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_visible)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_visible)

        # Both has error in the current data, and new data is being loaded
        # (due to difference in n_clicks and nr_loads) - text should be hidden
        error_data_different_clicks_and_loads_curr_visible_result = \
            update_run_graph_visibility(1, self.error_data, 0, self.visible_style)
        error_data_different_clicks_and_loads_curr_hidden_result = \
            update_run_graph_visibility(1, self.error_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(error_data_different_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(error_data_different_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # The data is loaded, but has an error - text should be hidden
        error_data_same_clicks_and_loads_curr_visible_result = \
            update_run_graph_visibility(0, self.error_data, 0, self.visible_style)
        error_data_same_clicks_and_loads_curr_hidden_result = \
            update_run_graph_visibility(0, self.error_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(error_data_same_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(error_data_same_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

    def test_update_run_graph(self):
        # If there is data error or data is loading or the selected metric is empty
        # the returned figure should be empty dict and there should be nothing loaded

        # Call with loading data.
        result_json = update_run_graph(1, LOSS_KEY, "{}", 0)
        figure, loaded = deserialize_update_run_graph_result(result_json)
        self.assertIsInstance(figure, dict)
        self.assertDictEqual(figure, {})
        self.assertIsInstance(loaded, str)
        self.assertEqual(loaded, EMPTY_LOADED)

        # Call with empty selection.
        result_json = update_run_graph(1, EMPTY_SELECTION, "{}", 1)
        figure, loaded = deserialize_update_run_graph_result(result_json)
        self.assertIsInstance(figure, dict)
        self.assertDictEqual(figure, {})
        self.assertIsInstance(loaded, str)
        self.assertEqual(loaded, EMPTY_LOADED)

        # Call with error data.
        result_json = update_run_graph(1, LOSS_KEY, '{"%s": "%s"}' % (ERROR_KEY, ERROR_MESSAGE), 1)
        figure, loaded = deserialize_update_run_graph_result(result_json)
        self.assertIsInstance(figure, dict)
        self.assertDictEqual(figure, {})
        self.assertIsInstance(loaded, str)
        self.assertEqual(loaded, EMPTY_LOADED)

        # Call with actual data
        run_data_json = json.dumps(self.loss_data)
        result_json = update_run_graph(1, LOSS_KEY, run_data_json, 1)
        figure, loaded = deserialize_update_run_graph_result(result_json)

        # Check that the loaded value corresponds to the passed metric and that
        # the figure contains the data from run_data_json
        self.assertIsInstance(loaded, str)
        self.assertEqual(loaded, LOSS_KEY)
        self.assertIsNotNone(figure)
        self.assertIsInstance(figure, dict)
        self.assertTrue(DATA_KEY in figure.keys())
        self.assertTrue('layout' in figure.keys())
        self.assertEqual(len(figure[DATA_KEY]), 1)
        self.assertEqual(figure[DATA_KEY][0]['x'], self.loss_data[LOSS_KEY][0]['x'])
        self.assertEqual(figure[DATA_KEY][0]['y'], self.loss_data[LOSS_KEY][0]['y'])
        self.assertEqual(figure[DATA_KEY][0]['name'], self.loss_data[LOSS_KEY][0]['name'])
        self.assertEqual(figure[DATA_KEY][0]['mode'], 'lines')
