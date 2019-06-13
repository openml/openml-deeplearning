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
    LOADING_TEXT_RUN_INFO,
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
        pass

    def test_update_flow_graph_text(self):
        pass

    def test_init_flow_loading(self):
        pass

    def test_update_flow_loading_info(self):
        pass

    def test_load_flow(self):
        pass

    def test_update_flow_error_text(self):
        pass

    def test_update_flow_graph_visibility(self):
        pass

    def test_update_flow_graph(self):
        pass
