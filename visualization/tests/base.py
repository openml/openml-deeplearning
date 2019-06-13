import unittest
import plotly.graph_objs as go

ERROR_KEY = 'error'
ERROR_MESSAGE = 'message'
DISPLAY_KEY = 'display'
MEAN_SQUARE_ERROR = 'mse'
MEAN_ABSOLUTE_ERROR = 'mae'


class VisualizationTestBase(unittest.TestCase):

    def setUp(self):
        self.run_id = 1

        self.error_data = '{"%s": "%s"}' % (ERROR_KEY, ERROR_MESSAGE)
        self.empty_data = '{}'
        self.simple_data = '{"run_id": %s}' % self.run_id
        self.non_empty_graph_data = go.Scatter(x=[1], y=[1], mode='lines', name='data')
        self.none_data = None

        self.non_empty_loading = ['load']
        self.empty_loading = []

        self.non_empty_error = ['error']
        self.empty_error = []

        self.display_hidden = 'none'
        self.display_visible = ''

        self.visible_style = {DISPLAY_KEY: ''}
        self.hidden_style = {DISPLAY_KEY: 'none'}
