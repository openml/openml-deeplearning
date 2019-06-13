from visualization.tests.base import (
    VisualizationTestBase,
    MEAN_SQUARE_ERROR,
    MEAN_ABSOLUTE_ERROR,
    ERROR_MESSAGE,
    ERROR_KEY,
    DISPLAY_KEY
)
from visualization.visualizer import (
    update_flow_info_texts_visibility,
    update_flow_graph_text,
    init_flow_loading,
    update_flow_loading_info,
    load_flow,
    update_flow_error_text,
    update_flow_graph_visibility,
    update_flow_graph,
    LOADING_TEXT_GENERAL,
    LOADING_TEXT_FLOW_INFO,
    FLOW_GRAPH_TEXT_TEMPLATE,
    METRIC_TO_LABEL
)
from visualization.tests.utils import (
    deserialize_style,
    deserialize_text_result,
    deserialize_loading_info_result,
    deserialize_id_loads_result,
    deserialize_info_texts_visibility_result
)


class TestVisualizationFlow(VisualizationTestBase):
    def setUp(self):
        super().setUp()

    def test_update_flow_info_texts_visibility(self):
        # There are no errors and nothing is loading so texts should be invisible
        result_json = update_flow_info_texts_visibility(self.empty_loading, self.empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are no errors, but flow data is loading, so loading text should be visible
        result_json = update_flow_info_texts_visibility(self.non_empty_loading, self.empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_hidden)

        # There are errors and no flow data is loading, so error text should be visible
        # and loading text should be hidden
        result_json = update_flow_info_texts_visibility(self.empty_loading, self.non_empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_hidden)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

        # There are errors and flow data is loading, so both texts should be visible
        result_json = \
            update_flow_info_texts_visibility(self.non_empty_loading, self.non_empty_error)
        load_style, error_style = deserialize_info_texts_visibility_result(result_json)
        self.assertEqual(load_style[DISPLAY_KEY], self.display_visible)
        self.assertEqual(error_style[DISPLAY_KEY], self.display_visible)

    def test_update_flow_graph_text(self):
        pass

    def test_init_flow_loading(self):
        # The function is used to pass data along, so the result should be same as input params

        # Tests with no flow id provided
        for n_clicks in range(10):
            no_flow_id_result = init_flow_loading(n_clicks, None)
            id_result, loads_result = deserialize_id_loads_result(no_flow_id_result)

            self.assertEqual(id_result, None)
            self.assertEqual(loads_result, n_clicks)

        # Test with flow id provided
        for n_clicks in range(10):
            flow_id_result = init_flow_loading(n_clicks, self.flow_id)
            id_result, loads_result = deserialize_id_loads_result(flow_id_result)

            self.assertEqual(id_result, self.flow_id)
            self.assertEqual(loads_result, n_clicks)

    def test_update_flow_loading_info(self):
        # There is no flow id, so flow data is not being loaded
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_but_no_id_result = \
            update_flow_loading_info(1, self.simple_data, None, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_but_no_id_result)
        self.assertEqual(result, self.empty_loading)

        # Load button has been clicked and the flow data is still not loaded
        # (but there is a flow id, which indicates it is being loaded)
        # (data param is used to trigger the callback and does not influence result)
        more_clicks_than_loaded_and_id_result = update_flow_loading_info(1, self.simple_data, 1, 0)
        result = deserialize_loading_info_result(more_clicks_than_loaded_and_id_result)
        self.assertEqual(result, self.non_empty_loading)

    def test_load_flow(self):
        pass

    def test_update_flow_error_text(self):
        # There is no flow data, so error message should be empty string
        no_data_result = update_flow_error_text(self.none_data)
        result = deserialize_text_result(no_data_result)
        self.assertEqual(result, '')

        # Flow data is empty, so error text should be empty string
        empty_data_result = update_flow_error_text(self.empty_data)
        result = deserialize_text_result(empty_data_result)
        self.assertEqual(result, '')

        # Flow data has error, so the text should be equal to the error message in the data
        error_data_result = update_flow_error_text(self.error_data)
        result = deserialize_text_result(error_data_result)
        self.assertEqual(result, ERROR_MESSAGE)

    def test_update_flow_graph_visibility(self):
        # Nothing is being displayed, so text should be hidden
        none_data_same_clicks_and_loads_currently_visible_result = \
            update_flow_graph_visibility(0, self.none_data, 0, self.visible_style)
        none_data_same_clicks_and_loads_currently_hidden_result = \
            update_flow_graph_visibility(0, self.none_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(none_data_same_clicks_and_loads_currently_visible_result)
        result_hidden = \
            deserialize_style(none_data_same_clicks_and_loads_currently_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # The load button has been clicked, but flow data has
        # not been loaded, so text should be hidden
        none_data_different_clicks_and_loads_1_curr_visible_result = \
            update_flow_graph_visibility(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_1_curr_hidden_result = \
            update_flow_graph_visibility(1, self.none_data, 0, self.hidden_style)
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
            update_flow_graph_visibility(1, self.none_data, 0, self.visible_style)
        none_data_different_clicks_and_loads_2_curr_hidden_result = \
            update_flow_graph_visibility(1, self.none_data, 0, self.hidden_style)
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
            update_flow_graph_visibility(1, self.empty_data, 0, self.visible_style)
        no_error_data_different_clicks_and_loads_curr_hidden_result = \
            update_flow_graph_visibility(1, self.empty_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(no_error_data_different_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(no_error_data_different_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # There is flow data without errors and the number of clicks and loads is the same so it's
        # neither loading, nor has error - text should be visible
        no_error_data_same_clicks_and_loads_curr_visible_result = \
            update_flow_graph_visibility(0, self.empty_data, 0, self.visible_style)
        no_error_data_same_clicks_and_loads_curr_hidden_result = \
            update_flow_graph_visibility(0, self.empty_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(no_error_data_same_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(no_error_data_same_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_visible)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_visible)

        # Both has error in the current flow data, and new flow data is being loaded
        # (due to difference in n_clicks and nr_loads) - text should be hidden
        error_data_different_clicks_and_loads_curr_visible_result = \
            update_flow_graph_visibility(1, self.error_data, 0, self.visible_style)
        error_data_different_clicks_and_loads_curr_hidden_result = \
            update_flow_graph_visibility(1, self.error_data, 0, self.hidden_style)
        result_visible = \
            deserialize_style(error_data_different_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(error_data_different_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

        # The flow data is loaded, but has an error - text should be hidden
        error_data_same_clicks_and_loads_curr_visible_result = \
            update_flow_graph_visibility(0, self.error_data, 0, self.visible_style)
        error_data_same_clicks_and_loads_curr_hidden_result = \
            update_flow_graph_visibility(0, self.error_data, 0, self.visible_style)
        result_visible = \
            deserialize_style(error_data_same_clicks_and_loads_curr_visible_result)
        result_hidden = \
            deserialize_style(error_data_same_clicks_and_loads_curr_hidden_result)
        self.assertEqual(result_visible[DISPLAY_KEY],
                         self.display_hidden)
        self.assertEqual(result_hidden[DISPLAY_KEY],
                         self.display_hidden)

    def test_update_flow_graph(self):
        pass
