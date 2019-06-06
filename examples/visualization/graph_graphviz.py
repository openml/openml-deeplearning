import os
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}


model = onnx.load_model('model.onnx')
pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir=".",
                            node_producer=GetOpNodeProducer(embed_docstring=True, **OP_STYLE))
pydot_graph.write_dot("graph.dot")
os.system('dot -Tsvg graph.dot -o graph.svg')

