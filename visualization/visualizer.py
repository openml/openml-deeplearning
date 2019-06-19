import os
import json
from google.protobuf import json_format

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import openml
from openml.exceptions import OpenMLServerException
from openml.tasks import OpenMLRegressionTask, OpenMLClassificationTask

import keras
import onnx
import torch
import mxnet as mx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

from visualization.utils import (
    has_error_or_is_loading,
    get_info_text_styles,
    get_loading_info,
    get_error_text,
    get_visibility_style,
    create_figure,
    extract_run_graph_data,
    get_training_data,
    get_onnx_model,
    simplyfy_pydot_graph,
    add_lists_element_wise
)

from visualization.constants import (
    DISPLAY_NONE,
    ERROR_KEY,
    LOADING_TEXT_RUN_INFO,
    LOADING_TEXT_FLOW_INFO,
    LOADING_TEXT_GENERAL,
    LOADING_TEXT_FLOW_GRAPH,
    METRIC_TO_LABEL,
    FLOW_GRAPH_TEXT_TEMPLATE,
    RUN_ID_KEY,
    RUN_GRAPH_TEXT_TEMPLATE,
    EMPTY_TEXT,
    GRAPH_EXPORT_STYLE,
    ONNX_MODEL_KEY,
    TASK_ID_KEY,
    TRAINING_DATA_KEY,
    TRAINING_DATA_URL_FORMAT,
    EMPTY_LOADED,
    EMPTY_SELECTION,
    STATIC_PATH,
    FLOW_ID_KEY,
    ACCURACY_KEY,
    MEAN_SQUARE_ERROR_KEY,
    MEAN_ABSOLUTE_ERROR_KEY,
    ROOT_MEAN_SQUARE_ERROR_KEY,
    LOSS_KEY
)

# Create the static folder if id does not exist
if not os.path.exists(STATIC_PATH):
    os.mkdir(STATIC_PATH)

# Use the external Dash stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Flow and run visualization', style={'text-align': 'center'}),  # HTML title
    html.Div(children=[
        dcc.Input(id='run-id', placeholder='Enter run id', type='number'),
        html.Button(id='load-run-button', n_clicks=0, children='Load run', style={'margin':
                                                                                  '0 10px'}),
        dcc.Input(id='flow-id', placeholder='Enter flow id', type='number'),
        html.Button(id='load-flow-button', n_clicks=0, children='Load flow', style={'margin':
                                                                                    '0 10px'}),
        html.Div(id='nr-run-loads', children='0', style={'display': DISPLAY_NONE}),
        html.Div(id='nr-flow-loads', children='0', style={'display': DISPLAY_NONE})
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
        html.Div(id='run-data'),
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
            options=[{'label': ERROR_KEY, 'value': ERROR_KEY}],
            values=[]),
        dcc.Checklist(
            id='error-flow-check',
            options=[{'label': ERROR_KEY, 'value': ERROR_KEY}],
            values=[]),
        html.Div(
            id='loaded-run-metric'),
        html.Div(
            id='loaded-flow-id'),
    ], style={'display': DISPLAY_NONE}),  # Hidden HTML elements used to transfer data
    html.Div(id='run-graph-div', children=[
        html.H3(id='run-graph-text',
                children=LOADING_TEXT_GENERAL,
                style={'text-align': 'center', 'margin': '30px 0 0 0'}),
        dcc.Dropdown(
            id='run-metric-dropdown',
            options=[],
            value='',
            clearable=False,
            style={'margin': '20px 0'}
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


# Define callback for retrieving files from the static folder
@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)


####################################################################################################
# Callbacks for runs
####################################################################################################

@app.callback([Output('info-run-loading-text', 'style'),
               Output('info-run-error-text', 'style')],
              [Input('load-run-check', 'values'),
               Input('error-run-check', 'values')])
def update_run_info_texts_visibility(load_values, error_values):
    return get_info_text_styles(load_values, error_values)


@app.callback(Output('run-graph-text', 'children'),
              [Input('run-metric-dropdown', 'value'),
               Input('loaded-run-metric', 'children')],
              [State('run-data', 'children')])
def update_run_graph_text(metric, loaded_metric, run_data_json):
    if run_data_json is None or metric == EMPTY_TEXT:  # There is no data
        return EMPTY_TEXT

    run_data = json.loads(run_data_json)

    if metric != loaded_metric:
        return LOADING_TEXT_RUN_INFO
    else:
        return RUN_GRAPH_TEXT_TEMPLATE.format(METRIC_TO_LABEL[metric], run_data[RUN_ID_KEY])


@app.callback([Output('run-id-info', 'children'),
               Output('nr-run-loads', 'children')],
              [Input('load-run-button', 'n_clicks')],
              [State('run-id', 'value')])
def init_run_loading(n_clicks, run_id):
    return run_id, n_clicks


@app.callback(Output('load-run-check', 'values'),
              [Input('load-run-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('run-id', 'value'),
               State('nr-run-loads', 'children')])
def update_run_loading_info(n_clicks, run_data_json, run_id, nr_loads):
    return get_loading_info(n_clicks, run_id, nr_loads)


@app.callback([Output('run-data', 'children'),
               Output('error-run-check', 'values'),
               Output('run-metric-dropdown', 'options'),
               Output('run-metric-dropdown', 'value')],
              [Input('run-id-info', 'children')])
def load_run(run_id):
    if run_id is None:
        return None, [], [], EMPTY_SELECTION

    try:
        run = openml.runs.get_run(run_id)

        # Check if the run contains training data and display an error if it does not
        if TRAINING_DATA_KEY not in run.output_files.keys():
            return json.dumps({ERROR_KEY: 'Run does not contain training information.'}), \
                [ERROR_KEY], [], EMPTY_SELECTION

        # Generate the url for the training data file
        data_url = TRAINING_DATA_URL_FORMAT.format(run.output_files[TRAINING_DATA_KEY])

        # Obtain the training data
        df = get_training_data(data_url, run.run_id)

        task = openml.tasks.get_task(run.task_id)
    except (OpenMLServerException, ValueError):
        return json.dumps({ERROR_KEY: 'There was an error retrieving the run.'}), \
            [ERROR_KEY], [], EMPTY_SELECTION

    if not isinstance(task, (OpenMLClassificationTask, OpenMLRegressionTask)):
        return json.dumps({ERROR_KEY: 'Associated task must be classification or '
                                      'regression.'}), \
            [ERROR_KEY], [], EMPTY_SELECTION

    # Obtain column names of the data to decide which metrics can be visualized
    df_columns = list(df.columns.values)

    metrics = [LOSS_KEY]

    # Add metrics present in the data to the list of metrics
    if isinstance(task, OpenMLClassificationTask):
        if ACCURACY_KEY in df_columns:
            metrics.append(ACCURACY_KEY)
    if isinstance(task, OpenMLRegressionTask):
        if MEAN_SQUARE_ERROR_KEY in df_columns:
            metrics.append(MEAN_SQUARE_ERROR_KEY)  # mean square error
        if MEAN_ABSOLUTE_ERROR_KEY in df_columns:
            metrics.append(MEAN_ABSOLUTE_ERROR_KEY)  # mean absolute error
        if ROOT_MEAN_SQUARE_ERROR_KEY in df_columns:
            metrics.append(ROOT_MEAN_SQUARE_ERROR_KEY)  # root mean square error

    folds = df['foldn'].max() + 1
    repn = df['repn'].max() + 1
    data = {
        RUN_ID_KEY: run_id,
        TASK_ID_KEY: task.task_id
    }
    dropdown_options = []
    means = {}
    fold_rep_count = 0

    for metric in metrics:
        # Create keys for each metric in the parsed data
        data[metric] = []
        # Create a dictionary for each metric in which means will be computed
        means[metric] = {
            'x': None,
            'sum': []
        }
        # Create the options for the dropdown menu
        dropdown_options.append({'label': METRIC_TO_LABEL[metric], 'value': metric})

    iter_per_epoch = len(df['iter'].unique())

    # Split the folds and repeats data
    for i in range(folds):
        for j in range(repn):
            # Counter is incremented and later used to compute mean value across folds and reps
            fold_rep_count += 1

            # Format the name of the line in the graph
            name = 'fold_{}_rep_{}'.format(i, j)

            # Filter the data for the given fold and rep
            fold_rep_data = \
                df[(df['foldn'] == i) & (df['repn'] == j)]

            x = [row['epoch'] * iter_per_epoch + row['iter']
                 for index, row in fold_rep_data.iterrows()]

            for metric in metrics:
                y = fold_rep_data[metric].tolist()

                # Add the current line to the sum for the given metric
                # Used to compute mean values
                means[metric]['sum'] = add_lists_element_wise(means[metric]['sum'], y)
                means[metric]['x'] = x

                # Add the parsed metric data to the list of data
                data[metric].append({
                    'x': x,
                    'y': y,
                    'name': name
                })

    for (metric, sum_dict) in means.items():
        y = [(x / fold_rep_count) for x in sum_dict['sum']]
        x = sum_dict['x']

        data[metric].append({
            'x': x,
            'y': y,
            'name': 'Mean'
        })

    return json.dumps(data), [], dropdown_options, LOSS_KEY


@app.callback(Output('info-run-error-text', 'children'),
              [Input('run-data', 'children')])
def update_run_error_text(run_data_json):
    return get_error_text(run_data_json)


@app.callback(Output('run-graph-div', 'style'),
              [Input('load-run-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('nr-run-loads', 'children'),
               State('run-graph-div', 'style')])
def update_run_graph_visibility(n_clicks, run_data_json, nr_loads, curr_style):
    return get_visibility_style(n_clicks, run_data_json, nr_loads, curr_style)


@app.callback([Output('run-graph', 'figure'),
               Output('loaded-run-metric', 'children')],
              [Input('load-run-button', 'n_clicks'),
               Input('run-metric-dropdown', 'value')],
              [State('run-data', 'children'),
               State('nr-run-loads', 'children')])
def update_run_graph(n_clicks, metric, run_data_json, nr_loads):
    if has_error_or_is_loading(n_clicks, run_data_json, nr_loads) or metric == EMPTY_SELECTION:
        return {}, EMPTY_LOADED

    run_data = json.loads(run_data_json)

    data = extract_run_graph_data(run_data, metric)

    return create_figure(data, METRIC_TO_LABEL[metric]), metric


####################################################################################################
# Callbacks for flows
####################################################################################################

@app.callback([Output('info-flow-loading-text', 'style'),
               Output('info-flow-error-text', 'style')],
              [Input('load-flow-check', 'values'),
               Input('error-flow-check', 'values')])
def update_flow_info_texts_visibility(load_values, error_values):
    return get_info_text_styles(load_values, error_values)


@app.callback(Output('flow-graph-text', 'children'),
              [Input('flow-id-info', 'children'),
               Input('loaded-flow-id', 'children')],
              [State('flow-data', 'children')])
def update_flow_graph_text(flow_id, loaded_id, flow_data_json):
    if flow_id is None:  # There is no data
        return EMPTY_TEXT

    if flow_data_json is None:
        return LOADING_TEXT_FLOW_INFO

    flow_data = json.loads(flow_data_json)

    if flow_id != loaded_id:
        return LOADING_TEXT_FLOW_INFO
    else:
        return FLOW_GRAPH_TEXT_TEMPLATE.format('Graph', flow_data[FLOW_ID_KEY])


@app.callback([Output('flow-id-info', 'children'),
               Output('nr-flow-loads', 'children')],
              [Input('load-flow-button', 'n_clicks')],
              [State('flow-id', 'value')])
def init_flow_loading(n_clicks, flow_id):
    return flow_id, n_clicks


@app.callback(Output('load-flow-check', 'values'),
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-data', 'children')],
              [State('flow-id', 'value'),
               State('nr-flow-loads', 'children')])
def update_flow_loading_info(n_clicks, flow_data_json, flow_id, nr_loads):
    return get_loading_info(n_clicks, flow_id, nr_loads)


@app.callback([Output('flow-data', 'children'),
               Output('error-flow-check', 'values')],
              [Input('flow-id-info', 'children')])
def load_flow(flow_id):
    if flow_id is None:
        return None, []

    try:
        flow = openml.flows.get_flow(flow_id, reinstantiate=True)
    except (OpenMLServerException, ValueError) as e:
        return json.dumps({ERROR_KEY: 'There was an error retrieving the flow.'.format(e)}), \
            [ERROR_KEY]

    if not isinstance(flow.model, (keras.models.Model, onnx.ModelProto, torch.nn.Module,
                                   mx.gluon.nn.HybridBlock)):
        return json.dumps({ERROR_KEY: 'The model corresponding to the flow cannot be '
                                      'visualized.'}), [ERROR_KEY]

    model = get_onnx_model(flow.model)

    if model is None:
        return json.dumps({ERROR_KEY: 'The flow could not be visualized.'}), [ERROR_KEY]

    model_dict = json_format.MessageToDict(model)

    return json.dumps({FLOW_ID_KEY: flow_id, ONNX_MODEL_KEY: model_dict}), []


@app.callback(Output('info-flow-error-text', 'children'),
              [Input('flow-data', 'children')])
def update_flow_error_text(flow_data_json):
    return get_error_text(flow_data_json)


@app.callback(Output('flow-graph-div', 'style'),
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-data', 'children')],
              [State('nr-flow-loads', 'children'),
               State('flow-graph-div', 'style')])
def update_flow_graph_visibility(n_clicks, flow_data_json, nr_loads, curr_style):
    return get_visibility_style(n_clicks, flow_data_json, nr_loads, curr_style)


@app.callback([Output('flow-graph', 'children'),
               Output('loaded-flow-id', 'children')],
              [Input('load-flow-button', 'n_clicks'),
               Input('flow-data', 'children')],
              [State('nr-flow-loads', 'children')])
def update_flow_graph(n_clicks, flow_data_json, nr_loads):
    if has_error_or_is_loading(n_clicks, flow_data_json, nr_loads):
        return None, EMPTY_LOADED

    flow_data = json.loads(flow_data_json)

    # Define paths
    dot_path = os.path.join(STATIC_PATH, 'graph_{}.dot'.format(flow_data[FLOW_ID_KEY]))
    svg_path = os.path.join(STATIC_PATH, 'graph_{}.svg'.format(flow_data[FLOW_ID_KEY]))

    # Recreate the passed onnx model from the dictionary
    model = onnx.ModelProto()
    json_format.ParseDict(flow_data[ONNX_MODEL_KEY], model)
    pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir=".",
                                node_producer=GetOpNodeProducer(
                                    embed_docstring=True, **GRAPH_EXPORT_STYLE))

    # Simplify the generated network graph
    pydot_graph = simplyfy_pydot_graph(pydot_graph)

    # Export the graph to a dot file and use it to create an svg
    pydot_graph.write_dot(dot_path)
    os.system('dot -Tsvg {} -o {}'.format(dot_path, svg_path))

    # Remove the unused dot file
    os.remove(dot_path)

    return html.Iframe(src='/static/graph_{}.svg'.format(flow_data[FLOW_ID_KEY]),
                       style={'width': '100%', 'height': '90vh'}), \
        flow_data[FLOW_ID_KEY]


# Run the Dash app server
if __name__ == '__main__':
    app.run_server(debug=True)
