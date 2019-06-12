import os
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import json
import plotly.graph_objs as go
import openml
from openml.tasks import OpenMLRegressionTask, OpenMLClassificationTask

ERROR_RUN_DATA_KEY = 'error'
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
    'acc': 'Accuracy'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Flow and run visualization', style={'text-align': 'center'}),  # HTML title
    html.Div(children=[
        dcc.Input(id='run-id', placeholder='Enter run id', type='number'),
        html.Button(id='load-run-button', n_clicks=0, children='Load run', style={'margin':
                                                                                  '0 10px'}),
        dcc.Input(id='flow-id', placeholder='Enter flow id', type='number'),
        html.Button(id='load-flow-button', n_clicks=0, children='Load flow', style={'margin':
                                                                                    '0 10px'}),
        html.Div(id='nr-run-clicks', children='0', style={'display': 'none'}),
        html.Div(id='nr-flow-clicks', children='0', style={'display': 'none'})
    ], style={'text-align': 'center'}),  # HTML elements for entering ids and load buttons
    html.Div(children=[
        html.H3(id='info-run-error-text', children='Error', style={'text-align': 'center'}),
        html.H3(id='info-run-loading-text', children=LOADING_TEXT_RUN_INFO,
                style={'text-align': 'center'}),
        html.H3(id='info-flow-error-text', children='Error', style={'text-align': 'center'}),
        html.H3(id='info-flow-loading-text', children=LOADING_TEXT_FLOW_INFO,
                style={'text-align': 'center'})
    ]),  # HTML elements for displaying errors and loading texts
    html.Div(children=[
        html.Div(id='run-id-info'),
        html.Div(id='flow-id-info'),
        html.Div(id='run-data', style={'display': 'none'}),  # TODO: Remove style
        html.Div(id='flow-data'),
        dcc.Checklist(
            id='load-flow-check',
            options=[{'label': 'load', 'value': 'load'}],
            values=[]),
        dcc.Checklist(
            id='load-run-check',
            options=[{'label': 'load', 'value': 'load'}],
            values=[]),
        dcc.Checklist(
            id='error-run-check',
            options=[{'label': 'error', 'value': 'error'}],
            values=[]),
        html.Div(
          id='loaded-run-metric'),
        dcc.Checklist(
            id='error-flow-check',
            options=[{'label': 'error', 'value': 'error'}],
            values=[]),
        dcc.Checklist(
            id='loaded-flow-check',
            options=[{'label': 'flow', 'value': 'flow'}],
            values=[]),
    ], style={'display': 'none'}),  # Hidden HTML elements used to transfer data  # TODO: change
    html.Div(id='run-graph-div', children=[
        html.H3(id='run-graph-text',
                children=LOADING_TEXT_GENERAL,
                style={'text-align': 'center', 'margin': '30px 0 0 0'}),
        dcc.Dropdown(
            id='run-metric-dropdown',
            options=[],
            value='',
            clearable=False,
            style={'margin': '20px 0'}  # TODO: Fix styling
        ),
        dcc.Graph(
            id='run-graph'
        ),
    ], style={'margin': '0 100px 0 100px'}),  # HTML elements for run information
    html.Div(id='flow-graph-div', children=[
        html.H3(id='flow-graph-text',
                children=LOADING_TEXT_FLOW_GRAPH,
                style={'text-align': 'center', 'margin': '30px 0 0 0'}),
        html.Div(id='flow-graph', style={'margin': '10px 100px 0 100px'})
    ], style={'margin': '0 100px 0 100px'})    # HTML elements for flow information
])


def has_error_or_is_loading(n_clicks, data_json, nr_clicks):
    if data_json is None or int(nr_clicks) < n_clicks:  # New run is being loaded
        return True

    data = json.loads(data_json)

    if ERROR_RUN_DATA_KEY in data.keys():
        return True

    return False


def get_info_text_styles(load_values, error_values):
    load_style = {'display': 'none', 'text-align': 'center'}
    error_style = {'display': 'none', 'text-align': 'center'}

    if len(load_values) != 0:
        load_style['display'] = ''

    if len(error_values) != 0:
        error_style['display'] = ''

    return load_style, error_style


def get_loading_info(n_clicks, item_id, nr_clicks):
    if int(nr_clicks) < n_clicks:  # User is loading new flow or run
        if item_id is None:
            return []

        return ['load']
    else:  # Data has finished loading
        return []


def get_error_text(data_json):
    if data_json is None:
        return ''

    data = json.loads(data_json)

    if ERROR_RUN_DATA_KEY in data.keys():
        return data[ERROR_RUN_DATA_KEY]

    return ''


def get_visibility_style(n_clicks, data_json, nr_clicks, curr_style):
    if has_error_or_is_loading(n_clicks, data_json, nr_clicks):
        curr_style['display'] = 'none'
    else:
        curr_style['display'] = ''

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


def extract_graph_data(run_data, key):
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


@app.callback([Output('info-run-loading-text', 'style'),
               Output('info-run-error-text', 'style')],
              [Input('load-run-check', 'values'),
               Input('error-run-check', 'values')])
def update_run_info_texts_visibility(load_values, error_values):
    return get_info_text_styles(load_values, error_values)


@app.callback([Output('info-flow-loading-text', 'style'),
               Output('info-flow-error-text', 'style')],
              [Input('load-flow-check', 'values'),
               Input('error-flow-check', 'values')])
def update_flow_info_texts_visibility(load_values, error_values):
    return get_info_text_styles(load_values, error_values)


@app.callback(Output('run-graph-text', 'children'),
              [Input('run-metric-dropdown', 'value'),
               Input('loaded-run-metric', 'children')],
              [State('run-data', 'children')])
def update_run_graph_texts(metric, loaded_metric, run_data_json):
    if run_data_json is None or metric == '':  # There is no data
        return ''

    run_data = json.loads(run_data_json)

    if metric != loaded_metric:
        return LOADING_TEXT_RUN_INFO
    else:
        return RUN_GRAPH_TEXT_TEMPLATE.format(METRIC_TO_LABEL[metric], run_data['run_id'])


@app.callback(Output('flow-graph-text', 'children'),
              [Input('load-flow-button', 'n_clicks'),
               Input('loaded-flow-check', 'values')],
              [State('nr-flow-clicks', 'children'),
               State('flow-data', 'children')])
def update_flow_text(n_clicks, loaded_values, nr_clicks, flow_data_json):
    if int(nr_clicks) < n_clicks or len(loaded_values) == 0:  # User is loading new flow
        return LOADING_TEXT_FLOW_GRAPH
    else:  # Data has finished loading
        flow_data = json.loads(flow_data_json)

        if 'flow' in loaded_values:
            graph_text = FLOW_GRAPH_TEXT_TEMPLATE.format('Graph', flow_data['flow_id'])
        else:
            graph_text = LOADING_TEXT_FLOW_GRAPH

        return graph_text


@app.callback(Output('loaded-flow-check', 'values'),
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-graph', 'children')],
              [State('nr-flow-clicks', 'children')])
def update_loaded_flow_check(n_clicks, network_graph, nr_clicks):
    if int(nr_clicks) < n_clicks:  # User is loading new flow
        return []
    else:  # Graph has finished loading
        loaded = []
        if network_graph is not None:
            loaded.append('flow')
        return loaded


@app.callback([Output('run-id-info', 'children'),
               Output('nr-run-clicks', 'children')],
              [Input('load-run-button', 'n_clicks')],
              [State('run-id', 'value')])
def init_run_loading(n_clicks, run_id):
    if run_id is None:
        return None, n_clicks

    return int(run_id), n_clicks


@app.callback([Output('flow-id-info', 'children'),
               Output('nr-flow-clicks', 'children')],
              [Input('load-flow-button', 'n_clicks')],
              [State('flow-id', 'value')])
def init_flow_loading(n_clicks, flow_id):
    if flow_id is None:
        return None, n_clicks

    return int(flow_id), n_clicks


@app.callback(Output('load-run-check', 'values'),
              [Input('load-run-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('run-id', 'value'),
               State('nr-run-clicks', 'children')])
def update_run_loading_info(n_clicks, run_data_json, run_id, nr_clicks):
    if int(nr_clicks) < n_clicks:  # User is loading new run
        if run_id is None:
            return []

        return ['load']
    else:  # Run data is loaded
        return []


@app.callback(Output('load-flow-check', 'values'),
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-data', 'children')],
              [State('flow-id', 'value'),
               State('nr-flow-clicks', 'children')])
def update_flow_loading_info(n_clicks, flow_data_json, flow_id, nr_clicks):
    return get_loading_info(n_clicks, flow_id, nr_clicks)


@app.callback([Output('run-data', 'children'),
               Output('error-run-check', 'values'),
               Output('run-metric-dropdown', 'options'),
               Output('run-metric-dropdown', 'value')],
              [Input('run-id-info', 'children')])
def load_run(run_id):
    if run_id is None:
        return None, [], [], ''

    try:
        run = openml.runs.get_run(run_id)
        flow = openml.flows.get_flow(run.flow_id)
        task = openml.tasks.get_task(run.task_id)
    except:
        return json.dumps({ERROR_RUN_DATA_KEY: 'There was an error retrieving the run.'}), \
               ['error'], [], ''

    if not isinstance(task, (OpenMLClassificationTask, OpenMLRegressionTask)):
        return json.dumps({ERROR_RUN_DATA_KEY: 'Associated task must be classification or '
                                               'regression.'}), \
               ['error'], [], ''

    # Read the data # TODO: Obtain actual training data file
    df = pd.read_csv('export.csv')

    metrics = ['loss']
    # TODO: Extract options from data instead
    if isinstance(task, OpenMLClassificationTask):
        metrics.append('acc')
    if isinstance(task, OpenMLRegressionTask):
        metrics.append('mse')  # mean square error
        metrics.append('mae')  # mean absolute error
        metrics.append('rmse')  # root mean square error

    folds = df['foldn'].max() + 1
    repn = df['repn'].max() + 1
    data = {
        'run_id': run_id,
        'flow_id': flow.flow_id,
        'task_id': task.task_id
    }
    dropdown_options = []

    for metric in metrics:
        # Create keys for each metric in the parsed data and simultaneously
        # create the options for the dropdown menu
        data[metric] = []
        dropdown_options.append({'label': METRIC_TO_LABEL[metric], 'value': metric})

    iter_per_epoch = len(df['iter'].unique())

    # Split the folds and repeats data
    for i in range(folds):
        for j in range(repn):
                name = 'fold_{}_rep_{}'.format(i, j)
                fold_rep_data = \
                    df[(df['foldn'] == i) & (df['repn'] == j)]

                x = [row['epoch'] * iter_per_epoch + row['iter']
                     for index, row in fold_rep_data.iterrows()]

                for metric in metrics:
                    data[metric].append({
                        'x': x,
                        'y': fold_rep_data[metric].tolist(),
                        'name': name
                    })

    return json.dumps(data), [], dropdown_options, 'loss'


@app.callback([Output('flow-data', 'children'),
               Output('error-flow-check', 'values')],
              [Input('flow-id-info', 'children')])
def load_flow(flow_id):
    if flow_id is None:
        return None, []

    # TODO: Retrieve flow info
    return json.dumps({'flow_id': flow_id}), []


@app.callback(Output('info-run-error-text', 'children'),
              [Input('run-data', 'children')])
def update_run_error_text(run_data_json):
    return get_error_text(run_data_json)


@app.callback(Output('info-flow-error-text', 'children'),
              [Input('flow-data', 'children')])
def update_flow_error_text(flow_data_json):
    return get_error_text(flow_data_json)


@app.callback(Output('run-graph-div', 'style'),
              [Input('load-run-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('nr-run-clicks', 'children'),
               State('run-graph-div', 'style')])
def update_run_graph_visibility(n_clicks, run_data_json, nr_clicks, curr_style):
    return get_visibility_style(n_clicks, run_data_json, nr_clicks, curr_style)


@app.callback(Output('flow-graph-div', 'style'),
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-data', 'children')],
              [State('nr-flow-clicks', 'children'),
               State('flow-graph-div', 'style')])
def update_flow_graph_visibility(n_clicks, flow_data_json, nr_clicks, curr_style):
    return get_visibility_style(n_clicks, flow_data_json, nr_clicks, curr_style)


@app.callback([Output('run-graph', 'figure'),
               Output('loaded-run-metric', 'children')],
              [Input('load-run-button', 'n_clicks'),
               Input('run-metric-dropdown', 'value')],
              [State('run-data', 'children'),
               State('nr-run-clicks', 'children')])
def update_run_graph(n_clicks, metric, run_data_json, nr_clicks):
    if run_data_json is None or metric == '' or int(nr_clicks) < n_clicks:
        # There is no data, no metric is selected, or user is loading a new run
        # The figure must be empty and nothing is loaded
        return {}, ''

    run_data = json.loads(run_data_json)

    if ERROR_RUN_DATA_KEY in run_data.keys():
        return {}, ''

    data = extract_graph_data(run_data, metric)

    return create_figure(data, METRIC_TO_LABEL[metric]), metric


@app.callback(Output('flow-graph', 'children'),
              [Input('flow-data', 'children')])
def update_flow_graph(flow_data_json):
    if flow_data_json is None:
        return None

    flow_data = json.loads(flow_data_json)

    if ERROR_RUN_DATA_KEY in flow_data.keys():
        return None

    # TODO: Actual graph and stuff
    return html.Iframe(src='/static/graph.svg', style={'width': '100%', 'height': '90vh'})


@app.server.route('/static/<resource>')
def serve_static(resource):
    STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return flask.send_from_directory(STATIC_PATH, resource)


if __name__ == '__main__':
    app.run_server(debug=True)
