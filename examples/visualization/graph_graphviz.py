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
for k, v in pydot_graph.obj_dict['nodes'].items():
    if "(op" in k:
        orig_name = v[0]['name']
        v[0]['name'] = orig_name.rsplit('\\n')[0] + '\"'
for k, v in pydot_graph.obj_dict['edges'].items():
    edge = v[0]
    lst = []
    for idx, rv in enumerate(edge['points']):
        if "(op" in rv:
            lst.append(rv.rsplit('\\n')[0] + '\"')
        else:
            lst.append(rv)
    edge['points'] = lst
    print(k, v)
pydot_graph.write_dot("graph.dot")
os.system('dot -Tsvg graph.dot -o graph.svg')

