from visualization.tests.base import (
    VisualizationTestBase,
    MEAN_SQUARE_ERROR,
    MEAN_ABSOLUTE_ERROR,
    ERROR_MESSAGE,
    DISPLAY_KEY
)
from visualization.visualizer import (
    update_run_info_texts_visibility,
    update_run_graph_text,
    init_run_loading,
    update_run_loading_info,
    load_run,
    update_run_error_text,
    update_run_graph_visibility,
    update_run_graph,
    LOADING_TEXT_GENERAL,
    LOADING_TEXT_RUN_INFO,
    RUN_GRAPH_TEXT_TEMPLATE,
    METRIC_TO_LABEL
)
from visualization.tests.utils import (
    deserialize_style,
    deserialize_text_result,
    deserialize_loading_info_result,
    deserialize_id_loads_result,
    deserialize_info_texts_visibility_result
)


class TestVisualizationRun(VisualizationTestBase):
    def setUp(self):
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
        # There is no id, so data is not being loaded
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_but_no_id_result = \
            update_run_loading_info(1, self.simple_data, None, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_but_no_id_result)
        self.assertEqual(result, self.empty_loading)

        # Load button has been clicked and the data is still not loaded
        # (but there is an id, which indicates it is being loaded)
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_and_id_result = update_run_loading_info(1, self.simple_data, 1, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_and_id_result)
        self.assertEqual(result, self.non_empty_loading)

    def test_load_run(self):
        pass

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
        pass
