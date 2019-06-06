import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import json
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Flow and run visualization', style={'text-align': 'center'}),

    dcc.Input(id='run-id', placeholder='Enter run id'),

    html.Button(id='load-button', n_clicks=0, children='Load'),

    dcc.Graph(
        id='loss-graph',
    ),

    dcc.Graph(
        id='acc-graph'
    ),

    html.Div(id='run-data', style={'display': 'none'})
])


@app.callback(Output('run-data', 'children'),
              [Input('load-button', 'n_clicks')],
              [State('run-id', 'value')])
def load_run(n_clicks, run_id):
    # Read the data
    df = pd.read_csv('export.csv')

    epochs = df['epoch'].max() + 1
    folds = df['foldn'].max() + 1
    repn = df['repn'].max() + 1
    data = {
        'loss': [],
        'acc': []
    }

    iter_per_epoch = len(df['iter'].unique())

    # Split the folds and repeats data
    for i in range(folds):
        for j in range(repn):
            for k in range(epochs):
                name = 'fold_{}_rep_{}_epoch_{}'.format(i, j, k)
                fold_rep_epoch_data = \
                    df[(df['foldn'] == i) & (df['repn'] == j) & (df['epoch'] == k)]

                x = [row['epoch'] * iter_per_epoch + row['iter']
                     for index, row in fold_rep_epoch_data.iterrows()]

                data['loss'].append({
                    'x': x,
                    'y': fold_rep_epoch_data['loss'].tolist(),
                    'name': name
                })

                data['acc'].append({
                    'x': x,
                    'y': fold_rep_epoch_data['acc'].tolist(),
                    'name': name
                })

    return json.dumps(data)


@app.callback(Output('loss-graph', 'figure'),
              [Input('run-data', 'children')])
def update_loss_graph(data_json):
    dataset = json.loads(data_json)
    data = []

    for item in dataset['loss']:
        data.append(
            go.Scatter(
                x=item['x'],
                y=item['y'],
                mode='lines',
                name=item['name']
            )
        )

    return {
            'data': data,
            'layout': go.Layout(
                xaxis={'title': 'Iterations'},
                yaxis={'title': 'Loss'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                # legend={'x': 0, 'y': 1},
                # hovermode='closest'
            )
        }

@app.callback(Output('acc-graph', 'figure'),
              [Input('run-data', 'children')])
def update_accuracy_graph(data_json):
    dataset = json.loads(data_json)
    data = []

    for item in dataset['acc']:
        data.append(
            go.Scatter(
                x=item['x'],
                y=item['y'],
                mode='lines',
                name=item['name']
            )
        )

    return {
            'data': data,
            'layout': go.Layout(
                xaxis={'title': 'Iterations'},
                yaxis={'title': 'Accuracy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                # legend={'x': 0, 'y': 1},
                # hovermode='closest'
            )
        }


if __name__ == '__main__':
    app.run_server(debug=True)
