import types
from typing import List, Optional, Tuple, Any, Union, Dict

import torch
import torch.nn as nn
import torchvision.models as tvmodels

import vizstack as vz
from vzlogger import connect, get_logger


class ComputationGraphNode:
    def __init__(self, payload: Any, variant: str = 'data'):
        self._payload = payload
        assert variant in ['module', 'fn', 'data']
        self._variant = variant

        self.parent = None
        self.ancestors = []
        self.alignments = []

    def set_ancestors(self, ancestors: List["ComputationGraphNode"]):
        if len(ancestors) > 0:
            self.parent = ancestors[-1]
            self.ancestors = list(ancestors)

    def has_ancestor(self, ancestor: "ComputationGraphNode"):
        return ancestor in self.ancestors

    def __view__(self):
        if self._variant == 'module':
            return vz.Token(self._payload.__class__.__name__)
        if self._variant == 'fn':
            return vz.Token(self._payload.fn.__name__)
        else:
            return vz.Token(self._payload.__class__.__name__)


class ComputationGraph:
    def __init__(self):
        self._stack: List[ComputationGraphNode] = []
        self._nodes: List[ComputationGraphNode] = []
        self._edges = []
        self._terminals = []

    def push(self, node: ComputationGraphNode) -> "ComputationGraph":
        node.set_ancestors(self._stack)
        self._stack.append(node)
        return self

    def pop(self) -> ComputationGraphNode:
        node = self._stack.pop()
        self._nodes.append(node)
        return node

    def edge(self, start: ComputationGraphNode, start_port: str, end: ComputationGraphNode, end_port: str):
        self._edges.append([start, start_port, end, end_port])

    def __view__(self):
        d = vz.Dag()
        node_to_id = {}
        for i, n in enumerate(self._nodes):
            d.node(str(i))
            d.item(n, str(i))
            node_to_id[n] = str(i)
        for n in self._nodes:
            if n.parent:
                d.node(node_to_id[n], parent=node_to_id[n.parent])
            for other in n.alignments:
                d.node(node_to_id[n], align_with={'axis': 'x', 'justify': 'north', 'nodes': [node_to_id[other]]})
        for edge in self._edges:
            d.port(node_to_id[edge[0]], edge[1], side='south' if edge[1].startswith('o') else 'north', order=int(edge[1][1:]))
            d.port(node_to_id[edge[2]], edge[3], side='south' if edge[3].startswith('o') else 'north', order=int(edge[3][1:]))
            d.edge({
                'id': node_to_id[edge[0]],
                'port': edge[1],
            }, {
                'id': node_to_id[edge[2]],
                'port': edge[3],
            })
        return d


class FunctionContext:
    def __init__(self, fn):
        self.fn = fn


class Tracker:
    def __init__(self, models: Optional[List[nn.Module]] = None):
        self._models = models if models is not None else []
        self._graph: Optional[ComputationGraph] = None
        self._old_modules = dict()

    def start(self) -> None:
        self._graph = ComputationGraph()
        self._old_modules['torch'] = dict()
        for var, value in vars(torch).items():
            if callable(value):
                setattr(torch, var, self._wrap_fn(value))
            self._old_modules['torch'][var] = value
        for model in self._models:
            for module in model.modules():
                module.register_forward_pre_hook(self._forward_pre_hook)
                module.register_forward_hook(self._forward_hook)

    def stop(self) -> ComputationGraph:
        graph = self._graph
        self._graph = None
        for var, value in self._old_modules['torch'].items():
            setattr(torch, var, value)
        # TODO: remove all hooks
        return graph

    def model(self, model: nn.Module):
        self._models.append(model)

    def _forward_pre_hook(self, m: Union[nn.Module, FunctionContext], inputs: Tuple[Any, ...]):
        node = ComputationGraphNode(m, 'module' if isinstance(m, nn.Module) else 'fn')
        # for each input, create a data node if it has no creator, then create an edge to the correct port
        setattr(m, '_cg_input_locations', [])
        for i, x in enumerate(inputs):
            if not isinstance(x, (list, tuple)):
                x = [x]
            locations = []
            for _x in x:
                try:
                    if not hasattr(_x, '_cg_location'):
                        # Test for primitives
                        setattr(_x, '_cg_location', None)
                        setattr(_x, '_cg_location', (self._graph.push(ComputationGraphNode(_x)).pop(), "o0"))
                    x_node, x_port = getattr(_x, '_cg_location')
                    locations.append((x_node, x_port))
                    self._graph.edge(x_node, x_port, node, f"i{i}")
                    setattr(_x, '_cg_location', (node, f"i{i}"))
                except AttributeError:
                    # `_x` is a builtin type and cannot have fields added to it
                    pass
            getattr(m, '_cg_input_locations').append(locations)
        self._graph.push(node)

    def _forward_hook(self, m: Union[nn.Module, FunctionContext], inputs: Tuple[Any], outputs: Any):
        node = self._graph.pop()
        for i, x in enumerate(inputs):
            if not isinstance(x, (list, tuple)):
                x = [x]
            locations = getattr(m, '_cg_input_locations')[i]
            for j, _x in enumerate(x):
                try:
                    # Test for primitives
                    setattr(_x, '_cg_location', None)
                except AttributeError:
                    continue
                x_node, x_port = locations[j]
                setattr(_x, '_cg_location', (x_node, x_port))

        # for each output, mark the node as its creator
        if not isinstance(outputs, tuple):
            outputs = [outputs]
        for i, y in enumerate(outputs):
            if not isinstance(y, (list, tuple)):
                y = [y]
            for _y in y:
                if hasattr(_y, '_cg_location') and getattr(_y, '_cg_location')[0].has_ancestor(node):
                    self._graph.edge(getattr(_y, '_cg_location')[0], getattr(_y, '_cg_location')[1], node, f"o{i}")
                setattr(_y, '_cg_location', (node, f"o{i}"))

        # for each param, check if it has a creator and add edge/alignment, then mark this as its creator
        if isinstance(m, nn.Module):
            if not hasattr(m, '_cg_nodes'):
                setattr(m, '_cg_nodes', [])
            for _node in getattr(m, '_cg_nodes'):
                if _node.parent == node.parent:
                    node.alignments.append(_node)
                    # create a container which wraps everything
            getattr(m, '_cg_nodes').append(node)

        # TODO: create data node for output if this is a terminal

# TODO: deal with in-place ops
# TODO: squeezenet no classifier

    def _wrap_fn(self, fn):
        def _wrapped(*args, **kwargs):
            ctx = FunctionContext(fn)
            inputs = args + tuple(kwargs.items())
            self._forward_pre_hook(ctx, inputs)
            outputs = fn(*args, **kwargs)
            self._forward_hook(ctx, inputs, outputs)
            return outputs

        return _wrapped


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.s = nn.Sequential(
            nn.ReLU(),
            nn.ELU(),
        )
        self.l2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.s(x)
        x = self.l1(x)
        return x


def main():
    model = Model()
    # model = tvmodels.squeezenet1_0()
    print(model)
    tracker = Tracker()
    tracker.model(model)
    tracker.start()
    # model(torch.rand((1, 3, 224, 224)))
    model(torch.rand(1, 128))
    graph = tracker.stop()
    with connect():
        get_logger('main').info(graph)
    print(vz.assemble(graph))


if __name__ == '__main__':
    main()
