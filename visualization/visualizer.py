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
LOADING_TEXT = 'Loading...'
GRAPH_TEXT_TEMPLATE = '{} for run {}'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Flow and run visualization', style={'text-align': 'center'}),
    html.Div(children=[
        dcc.Input(id='run-id', placeholder='Enter run id', type='number'),
        html.Button(id='load-button', n_clicks=0, children='Load', style={'margin-left': '10px'}),
        html.Div(id='nr-clicks', children='0', style={'display': 'none'})
    ], style={'text-align': 'center'}),
    html.Div(children=[
        html.H3(id='info-error-text', children='Error', style={'text-align': 'center'}),
        html.H3(id='info-loading-text', children=LOADING_TEXT, style={'text-align': 'center'})
    ]),
    html.Div(children=[
        html.Div(id='run-id-info'),
        html.Div(id='run-data'),
        dcc.Checklist(
            id='load-check',
            options=[{'label': 'load', 'value': 'load'}],
            values=[]),
        dcc.Checklist(
            id='error-check',
            options=[{'label': 'error', 'value': 'error'}],
            values=[]),
        dcc.Checklist(
            id='loaded-check',
            options=[{'label': 'loss', 'value': 'loss'},
                     {'label': 'acc', 'value': 'acc'},
                     {'label': 'graph', 'value': 'graph'}],
            values=[])
    ], style={'display': 'none'}),
    html.Div(children=[
        html.H3(id='loss-graph-text', children=LOADING_TEXT, style={'text-align': 'center',
                                                                    'margin': '30px 0 0 0'}),
        dcc.Graph(
            id='loss-graph'
        ),
        html.H3(id='acc-graph-text', children=LOADING_TEXT, style={'text-align': 'center',
                                                                   'margin': '30px 0 0 0'}),
        dcc.Graph(
            id='acc-graph'
        ),
    ], style={'margin': '0 100px 0 100px'}, id='graph-div'),
    html.Div(id='network_graph_div', children=[
        html.H3(id='network-graph-text', children=LOADING_TEXT, style={'text-align': 'center',
                                                                       'margin': '30px 0 0 0'}),
        html.Div(id='network-graph', style={'margin': '10px 100px 0 100px'})
    ], style={'display': 'none'})
])


@app.callback([Output('info-loading-text', 'style'),
               Output('info-error-text', 'style')],
              [Input('load-check', 'values'),
               Input('error-check', 'values')])
def update_info_texts_visibility(load_values, error_values):
    load_style = {'display': 'none', 'text-align': 'center'}
    error_style = {'display': 'none', 'text-align': 'center'}

    if len(load_values) != 0:
        load_style['display'] = ''

    if len(error_values) != 0:
        error_style['display'] = ''

    return load_style, error_style


@app.callback([Output('loss-graph-text', 'children'),
               Output('acc-graph-text', 'children'),
               Output('network-graph-text', 'children')],
              [Input('load-button', 'n_clicks'),
               Input('loaded-check', 'values')],
              [State('nr-clicks', 'children'),
               State('run-data', 'children')])
def update_graph_texts(n_clicks, loaded_values, nr_clicks, run_data_json):
    if int(nr_clicks) < n_clicks or len(loaded_values) == 0:  # User is loading new run
        return LOADING_TEXT, LOADING_TEXT, LOADING_TEXT
    else:  # Data has finished loading
        run_data = json.loads(run_data_json)

        if 'loss' in loaded_values:
            loss_text = GRAPH_TEXT_TEMPLATE.format('Loss', run_data['run_id'])
        else:
            loss_text = LOADING_TEXT

        if 'acc' in loaded_values:
            acc_text = GRAPH_TEXT_TEMPLATE.format('Accuracy', run_data['run_id'])
        else:
            acc_text = LOADING_TEXT

        if 'graph' in loaded_values:
            graph_text = GRAPH_TEXT_TEMPLATE.format('Graph of flow', run_data['run_id'])
        else:
            graph_text = LOADING_TEXT

        return loss_text, acc_text, graph_text


@app.callback(Output('loaded-check', 'values'),
              [Input('load-button', 'n_clicks'),
               Input('loss-graph', 'figure'),
               Input('acc-graph', 'figure'),
               Input('network-graph', 'children')],
              [State('nr-clicks', 'children')])
def update_loaded_check(n_clicks, loss_figure, acc_figure, network_graph, nr_clicks):
    if int(nr_clicks) < n_clicks or loss_figure == {}:  # User is loading new run
        return []
    else:  # Graph has finished loading
        loaded = []
        if loss_figure != {}:
            loaded.append('loss')
        if acc_figure != {}:
            loaded.append('acc')
        if network_graph is not None:
            loaded.append('graph')
        return loaded


@app.callback([Output('run-id-info', 'children'),
               Output('nr-clicks', 'children')],
              [Input('load-button', 'n_clicks')],
              [State('run-id', 'value')])
def init_loading(n_clicks, run_id):
    if run_id is None:
        return None, n_clicks

    return int(run_id), n_clicks


@app.callback(Output('load-check', 'values'),
              [Input('load-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('run-id', 'value'),
               State('nr-clicks', 'children')])
def update_loading_info(n_clicks, run_data_json, run_id, nr_clicks):
    if int(nr_clicks) < n_clicks:  # User is loading new run
        if run_id is None:
            return []

        return ['load']
    else:  # Data has finished loading
        return []


@app.callback([Output('run-data', 'children'),
               Output('error-check', 'values')],
              [Input('run-id-info', 'children')])
def load_run(run_id):
    if run_id is None:
        return None, []

    try:
        run = openml.runs.get_run(run_id)
        flow = openml.flows.get_flow(run.flow_id)
        task = openml.tasks.get_task(run.task_id)
    except:
        return json.dumps({ERROR_RUN_DATA_KEY: 'There was an error retrieving the run.'}), ['error']

    if not isinstance(task, (OpenMLClassificationTask, OpenMLRegressionTask)):
        return json.dumps({ERROR_RUN_DATA_KEY: 'Associated task must be classification or '
                                               'regression.'}), ['error']

    # Read the data # TODO: Obtain actual training data file
    df = pd.read_csv('export.csv')

    folds = df['foldn'].max() + 1
    repn = df['repn'].max() + 1
    data = {
        'loss': [],
        'acc': [],
        'run_id': run_id,
        'flow_id': flow.flow_id,
        'task_id': task.task_id
    }

    iter_per_epoch = len(df['iter'].unique())

    # Split the folds and repeats data
    for i in range(folds):
        for j in range(repn):
                name = 'fold_{}_rep_{}'.format(i, j)
                fold_rep_data = \
                    df[(df['foldn'] == i) & (df['repn'] == j)]

                x = [row['epoch'] * iter_per_epoch + row['iter']
                     for index, row in fold_rep_data.iterrows()]

                data['loss'].append({
                    'x': x,
                    'y': fold_rep_data['loss'].tolist(),
                    'name': name
                })

                data['acc'].append({
                    'x': x,
                    'y': fold_rep_data['acc'].tolist(),
                    'name': name
                })

    return json.dumps(data), []


@app.callback(Output('info-error-text', 'children'),
              [Input('run-data', 'children')])
def update_error_text(run_data_json):
    if run_data_json is None:
        return ''

    run_data = json.loads(run_data_json)

    if ERROR_RUN_DATA_KEY in run_data.keys():
        return run_data[ERROR_RUN_DATA_KEY]

    return ''


@app.callback([Output('loss-graph-text', 'style'),
               Output('loss-graph', 'style'),
               Output('acc-graph-text', 'style'),
               Output('acc-graph', 'style'),
               Output('network_graph_div', 'style')],
              [Input('load-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('nr-clicks', 'children')])
def update_graph_visibility(n_clicks, run_data_json, nr_clicks):
    default_graph_style = {'display': 'none', 'text-align': 'center'}
    default_text_style = {'text-align': 'center', 'margin': '50px 0 0 0', 'display': 'none'}

    if run_data_json is None or int(nr_clicks) < n_clicks:  # New run is being loaded
        return default_text_style, default_graph_style, \
               default_text_style, default_graph_style, default_graph_style

    run_data = json.loads(run_data_json)

    if ERROR_RUN_DATA_KEY in run_data.keys():
        return default_text_style, default_graph_style, \
               default_text_style, default_graph_style, default_graph_style

    default_graph_style['display'] = ''
    default_text_style['display'] = ''

    return default_text_style, default_graph_style, \
        default_text_style, default_graph_style, default_graph_style


@app.callback(Output('loss-graph', 'figure'),
              [Input('run-data', 'children')])
def update_loss_graph(run_data_json):
    if run_data_json is None:
        return {}

    run_data = json.loads(run_data_json)

    if ERROR_RUN_DATA_KEY in run_data.keys():
        return {}

    data = extract_graph_data(run_data, 'loss')

    return create_figure(data, 'Loss')


@app.callback(Output('acc-graph', 'figure'),
              [Input('run-data', 'children')])
def update_accuracy_graph(run_data_json):
    if run_data_json is None:
        return {}

    run_data = json.loads(run_data_json)

    if ERROR_RUN_DATA_KEY in run_data.keys():
        return {}

    data = extract_graph_data(run_data, 'acc')

    return create_figure(data, 'Accuracy')


@app.callback(Output('network-graph', 'children'),
              [Input('load-button', 'n_clicks'),
               Input('run-data', 'children')],
              [State('nr-clicks', 'children')])
def update_network_graph(n_clicks, run_data_json, nr_clicks):
    if int(nr_clicks) < n_clicks:
        return None

    # TODO: Export graph
    return html.Iframe(src='/static/graph.svg', style={'width': '100%', 'height': '90vh'})


@app.server.route('/static/<resource>')
def serve_static(resource):
    STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return flask.send_from_directory(STATIC_PATH, resource)


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


if __name__ == '__main__':
    app.run_server(debug=True)
