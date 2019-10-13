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
            return vz.Token(self._payload.__name__)
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
        # if there's already an edge from start to a descendant of end, link that edge to end instead of start
        for edge in self._edges:
            if edge[0] == start and edge[1] == start_port and edge[2].has_ancestor(end):
                edge[0] = end
                edge[1] = end_port
                # if the edge would now be routed from child to parent to another child, move the child out to a higher
                # level until that's not true
                while start.has_ancestor(end):
                    start.ancestors.pop()
                    start.parent = start.ancestors[-1] if len(start.ancestors) > 0 else None
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
        for edge in self._edges:
            d.port(node_to_id[edge[0]], edge[1], side='south' if edge[1].startswith('o') else 'north')
            d.port(node_to_id[edge[2]], edge[3], side='south' if edge[3].startswith('o') else 'north')
            d.edge({
                'id': node_to_id[edge[0]],
                'port': edge[1],
            }, {
                'id': node_to_id[edge[2]],
                'port': edge[3],
            })
        return d
        # for start, start_port, end, end_port in self._edges:
        #     print(start._payload.__class__.__name__, start_port, '->')
        #     print(end._payload.__class__.__name__, end_port)


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

    def _forward_pre_hook(self, m: nn.Module, inputs: Tuple[Any, ...]):
        self._graph.push(ComputationGraphNode(m, 'module'))

    def _forward_hook(self, m: Optional[nn.Module], inputs: Tuple[Any], outputs: Any):
        node = self._graph.pop()
        # for each input, create a data node if it has no creator, then create an edge to the correct port
        for i, x in enumerate(inputs):
            x_nodes = []
            if not isinstance(x, (list, tuple)):
                x = [x]
            for _x in x:
                try:
                    if not hasattr(_x, '_cg_creator'):
                        setattr(_x, '_cg_creator', None)
                        setattr(_x, '_cg_creator', self._graph.push(ComputationGraphNode(_x)).pop())
                        setattr(_x, '_cg_creator_port', "o0")
                    x_nodes.append(getattr(_x, '_cg_creator'))
                    self._graph.edge(x_nodes[-1], getattr(_x, '_cg_creator_port'), node, f"i{i}")
                except AttributeError:
                    # `_x` is a builtin type and cannot have fields added to it
                    pass

        # for each output, mark the node as its creator
        if not isinstance(outputs, tuple):
            outputs = [outputs]
        for i, y in enumerate(outputs):
            if not isinstance(y, (list, tuple)):
                y = [y]
            for _y in y:
                if hasattr(_y, '_cg_creator') and getattr(_y, '_cg_creator').has_ancestor(node):
                    self._graph.edge(getattr(_y, '_cg_creator'), getattr(_y, '_cg_creator_port'), node, f"o{i}")
                setattr(_y, '_cg_creator', node)
                setattr(_y, '_cg_creator_port', f"o{i}")

        # for each param, check if it has a creator and add edge/alignment, then mark this as its creator
        if isinstance(m, nn.Module):
            pass
            # TODO

        # TODO: create data node for output if this is a terminal

    def _wrap_fn(self, f):
        def _wrapped(*args, **kwargs):
            print(f.__name__)
            self._graph.push(ComputationGraphNode(f, 'fn'))
            y = f(*args, **kwargs)
            self._forward_hook(None, args + tuple(kwargs.items()), y)
            return y

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
        x = torch.flatten(x)
        x = self.l2(x)
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
