import os

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ONNX_MODEL_PATH = os.path.join(STATIC_PATH, 'model.onnx')

TRAINING_DATA_URL_FORMAT = 'https://www.openml.org/data/download/{}/training.csv'

TRAINING_DATA_KEY = 'training'
RUN_ID_KEY = 'run_id'
FLOW_ID_KEY = 'flow_id'
TASK_ID_KEY = 'task_id'
ERROR_KEY = 'error'
ONNX_MODEL_KEY = 'model'
DISPLAY_NONE = 'none'
DISPLAY_VISIBLE = ''
EMPTY_TEXT = ''
EMPTY_LOADED = ''
EMPTY_SELECTION = ''
ACCURACY_KEY = 'accuracy'
MEAN_SQUARE_ERROR_KEY = 'mse'
MEAN_ABSOLUTE_ERROR_KEY = 'mae'
ROOT_MEAN_SQUARE_ERROR_KEY = 'rmse'
LOSS_KEY = 'loss'
DISPLAY_KEY = 'display'

LOADING_TEXT_FORMAT = 'Loading{}...'
LOADING_TEXT_GENERAL = LOADING_TEXT_FORMAT.format('')
LOADING_TEXT_RUN_INFO = LOADING_TEXT_FORMAT.format(' run information')
LOADING_TEXT_FLOW_INFO = LOADING_TEXT_FORMAT.format(' flow information')
LOADING_TEXT_FLOW_GRAPH = LOADING_TEXT_FORMAT.format(' flow graph')
RUN_GRAPH_TEXT_TEMPLATE = '{} for run {}'
FLOW_GRAPH_TEXT_TEMPLATE = '{} for flow {}'

METRIC_TO_LABEL = {
    'mse': 'Mean Square Error',
    'loss': 'Loss',
    'mae': 'Mean Absolute Error',
    'rmse': 'Root Mean Square Error',
    'accuracy': 'Accuracy'
}

GRAPH_EXPORT_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}
