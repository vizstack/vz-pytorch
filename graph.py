import types
import inspect
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Union, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

import vizstack as vz
from vzlogger import connect, get_logger


class ComputationGraphNode:
    def __init__(self, payload: Any, variant: str):
        self._payload = payload
        assert variant in ["call", "data"]
        self._variant = variant
        if self._variant == "call":
            assert isinstance(payload, (nn.Module, FunctionContext))

        self.parent = None
        self.ancestors = []
        self.alignments = []
        self.temporal_groups = [[]]

        self.edges: List[Tuple[str, ComputationGraphNode, str]] = []

        self.port_names = defaultdict(lambda: None)
        if self._variant == "call":
            fn = payload.forward if isinstance(payload, nn.Module) else payload.fn
            try:
                for i, param in enumerate(
                    filter(
                        lambda p: not isinstance(payload, nn.Module) or p != "self", inspect.signature(fn).parameters
                    )
                ):
                    self.port_names[f"i{i}"] = param
            except ValueError:
                pass

        # Create the fragment assembler immediately in case it mutates later
        if isinstance(self._payload, nn.Module):
            self._view = vz.Token(self._payload.__class__.__name__, color='purple')
        elif isinstance(self._payload, torch.Tensor):
            self._view = vz.Token(f"Tensor{list(self._payload.shape)}", color='blue')
        else:
            self._view = vz.view(self._payload)

    def child(self, node: "ComputationGraphNode"):
        node.ancestors = self.ancestors + [self]
        node.parent = self
        self.temporal_groups[-1].append(node)
        return node

    def edge(
        self, start_port: str, end: "ComputationGraphNode", end_port: str,
    ):
        self.edges.append((start_port, end, end_port))

    def has_ancestor(self, ancestor: "ComputationGraphNode"):
        return ancestor in self.ancestors

    def is_expanded(self):
        return not isinstance(self._payload, nn.Module) or "torch.nn" not in self._payload.__module__

    def create_temporal(self):
        if len(self.temporal_groups[-1]) > 0:
            self.temporal_groups.append([])

    def __view__(self):
        return self._view


class FunctionContext:
    def __init__(self, fn):
        self.fn = fn
        self.cg_input_locations: List[Tuple[ComputationGraphNode, str]] = []

    def __view__(self):
        return vz.Token(self.fn.__name__, color='red')


_MAGIC_METHODS = [
    "__add__",
    "__sub__",
    "__mul__",
    "__div__",
    "__floordiv__",
    "__truediv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__or__",
    "__xor__",
    "__getitem__",
]
_MAGIC_METHODS += ["__r" + fn_name.lstrip("__") for fn_name in _MAGIC_METHODS]


class Tracker:
    def __init__(self, models: Optional[List[nn.Module]] = None):
        self._models = models if models is not None else []
        self._node: Optional[ComputationGraphNode] = None
        self._old_modules: Dict[str, Dict[str, Any]] = dict()

    def start(self) -> None:
        self._node = ComputationGraphNode(None, "data")
        self._old_modules["torch"] = dict()
        self._old_modules["torch.Tensor"] = dict()
        for var, value in vars(torch).items():
            if callable(value) and not inspect.isclass(value):
                setattr(torch, var, self._wrap_fn(value))
                self._old_modules["torch"][var] = value

        for var in dir(torch.Tensor):
            value = getattr(torch.Tensor, var)
            if callable(value) and var in _MAGIC_METHODS:
                setattr(torch.Tensor, var, self._wrap_fn(value))
                self._old_modules["torch.Tensor"][var] = value
        for model in self._models:
            for module in model.modules():
                module.register_forward_pre_hook(self._on_call)
                module.register_forward_hook(self._on_return)

    def finish(self):
        nodes = [(self._node, None)]
        self._node = None
        for var, value in self._old_modules["torch"].items():
            setattr(torch, var, value)
        for var, value in self._old_modules["torch.Tensor"].items():
            setattr(torch.Tensor, var, value)
        # TODO: remove all hooks

        d = vz.Dag()
        node_to_id = {}
        while len(nodes) > 0:
            node, parent = nodes.pop()
            node_to_id[node] = str(len(node_to_id))
            print(node_to_id[node], len(node.temporal_groups))
            d.node(
                node_to_id[node],
                parent=parent,
                item=node,
                is_expanded=node.is_expanded(),
                flow_direction="south" if len(node.temporal_groups) == 1 else "east",
            )
            if len(node.temporal_groups) > 1:
                for t in range(len(node.temporal_groups)):
                    d.node(
                        f"{node_to_id[node]}_{t}",
                        parent=node_to_id[node],
                        is_interactive=False,
                        flow_direction="south",
                    )
                    d.item(None, f"{node_to_id[node]}_{t}")
                    for child in node.temporal_groups[t]:
                        nodes.append((child, f"{node_to_id[node]}_{t}"))
            else:
                for child in node.temporal_groups[0]:
                    nodes.append((child, node_to_id[node]))
        for node in node_to_id:
            for edge in node.edges:
                start_port, end_node, end_port = edge
                if end_node not in node_to_id:
                    continue
                d.port(
                    node_to_id[node],
                    start_port,
                    side="south" if start_port.startswith("o") else "north",
                    order=int(start_port[1:]),
                    label=node.port_names[start_port],
                )
                d.port(
                    node_to_id[end_node],
                    end_port,
                    side="south" if end_port.startswith("o") else "north",
                    order=int(end_port[1:]),
                    label=end_node.port_names[end_port],
                )
                d.edge(
                    {"id": node_to_id[node], "port": start_port,}, {"id": node_to_id[end_node], "port": end_port,},
                )
        # TODO: alignments
        # for other in node.alignments:
        #     d.node(
        #         node_to_id[node], align_with={"axis": "x", "justify": "north", "nodes": [node_to_id[other]], },
        #     )
        d.node("0", is_visible=False, is_interactive=False)
        return d

    def model(self, model: nn.Module):
        self._models.append(model)

    def tick(self):
        self._node.create_temporal()

    def _on_call(self, fn: Union[nn.Module, FunctionContext], arguments: Tuple[Any, ...]):
        call_node = self._node.child(ComputationGraphNode(fn, "call"))

        argument_locations: List[List[Tuple[ComputationGraphNode, str]]] = []
        for i, arg in enumerate(arguments):
            # We do "iterable cracking" by default, so if the argument isn't an iterable already, make it one
            arg_iterable: Union[List[Any], Tuple[Any]]
            if not isinstance(arg, (list, tuple)):
                arg_iterable = (arg,)
            else:
                arg_iterable = arg
            arg_iterable_locations: List[Tuple[ComputationGraphNode, str]] = []
            for arg_value in arg_iterable:
                try:
                    # If the argument value has never been seen in the graph before, create a new node for it
                    if not hasattr(arg_value, "cg_location"):
                        # Test for primitives. If we don't try to set `cg_location` to a dummy value first, the new
                        # data node will be created before `cg_location` is set, causing there to be an extraneous
                        # data node connected to nothing.
                        # TODO: should we create data nodes for these primitives anyway?
                        arg_value.cg_location = None
                        arg_value.cg_location = (
                            self._node.child(ComputationGraphNode(arg_value, "data")),
                            "o0",
                        )
                    # Create an edge from the value's current location to `call_node`, set its new current location to
                    # be `call_node`, then cache its old location on `call_node` so that, when it returns, it can reset
                    # the value's location.
                    arg_iterable_locations.append(arg_value.cg_location)
                    start_node, start_port = arg_value.cg_location
                    start_node.edge(start_port, call_node, f"i{i}")
                    arg_value.cg_location = (call_node, f"i{i}")
                except AttributeError:
                    # `arg_value` is a primitive type and cannot have fields added to it
                    pass
            argument_locations.append(arg_iterable_locations)
        fn.cg_input_locations = argument_locations
        self._node = call_node

    def _on_return(self, fn: Union[nn.Module, FunctionContext], arguments: Tuple[Any], outputs: Any):
        print(self._node)
        for i, arg in enumerate(arguments):
            # We do "iterable cracking" by default, so if the argument isn't an iterable already, make it one
            arg_iterable: Union[List[Any], Tuple[Any]]
            if not isinstance(arg, (list, tuple)):
                arg_iterable = (arg,)
            else:
                arg_iterable = arg
            input_locations = fn.cg_input_locations[i]
            for j, arg_value in enumerate(arg_iterable):
                try:
                    # Test for primitives. If we don't set `cg_location` to a dummy value first, then `input_locations`
                    # will be indexed into at element `j` first, which will not exist if `arg_value` was a primitive and
                    # thus added no input location in `_on_call()`.
                    arg_value.cg_location = None
                    arg_value.cg_location = input_locations[j]
                except AttributeError:
                    continue

        # We do "iterable cracking" by default, so if there's only one output, make it a single-element tuple
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for i, output in enumerate(outputs):
            # We do "iterable cracking" by default, so if the output isn't an iterable already, make it one
            output_iterable: Union[List[Any], Tuple[Any]]
            if not isinstance(output, (list, tuple)):
                output_iterable = (output,)
            else:
                output_iterable = output
            for output_value in output_iterable:
                # If the output value has a current location in the graph and that location is a descendant of
                # `call_node`, then add an edge from that descendant to this node's output port
                if hasattr(output_value, "cg_location") and output_value.cg_location[0].has_ancestor(self._node):
                    start_node, start_port = output_value.cg_location
                    start_node.edge(start_port, self._node, f"o{i}")
                data_node = self._node.parent.child(ComputationGraphNode(output_value, "data"))
                self._node.edge(f"o{i}", data_node, "i0")
                output_value.cg_location = (data_node, "o0")
        self._node = self._node.parent

    # TODO: cleanup graph and node and separate out functions
    # TODO: primitive inputs (i.e., integers into __getitem__)
    # TODO: don't duplicate data nodes when they get passed through parent interface
    # TODO: deal with in-place ops (maybe done?)
    # TODO: make modules return subclassed primitives (why?)

    def _wrap_fn(self, fn):
        def _wrapped(*args, **kwargs):
            ctx = FunctionContext(fn)
            inputs = args + tuple(kwargs.items())
            self._on_call(ctx, inputs)
            outputs = fn(*args, **kwargs)
            is_output_iterable = isinstance(outputs, (list, tuple))
            if not is_output_iterable:
                outputs = (outputs,)
            has_tensor = any([isinstance(output, torch.Tensor) for output in outputs])
            if not is_output_iterable:
                outputs = outputs[0]
            self._on_return(ctx, inputs, outputs)
            return outputs

        return _wrapped


_tracker = None


def start(model):
    global _tracker
    assert _tracker is None
    _tracker = Tracker(models=[model])
    _tracker.start()


def finish():
    global _tracker
    assert _tracker is not None
    return _tracker.finish()


def tick():
    global _tracker
    assert _tracker is not None
    _tracker.tick()


# Whenever a module gets called:
# PRE-HOOK
# 1. create a node with the module as payload
# 2. if the current parent on the stack has a child with the same payload, create a new temporal container
# 3. for each input, check if it has a location (a node-port pair);
#   if not, create a new data node for it and make that its location
#   then, cache old location on the module and set its location to the input port of the module node, creating an edge
# 4. set module ancestors and add it to the stack
# POST-HOOK
# 1. pop the node off of the stack
# 2. for each input, set its location back to the cached location
# OLD 3. for each output, set its location to the module's output port
# NEW 3. for each output, create a data node and set its location to the data node


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.s = nn.Sequential(nn.ReLU(), nn.ELU(),)
        self.l2 = nn.Linear(128, 128)

    def forward(self, x):
        tick()
        x = self.l1(x)
        x = F.relu(x)
        tick()
        x = self.l1(x)
        x = F.relu(x)
        tick()
        x = self.l1(x)
        x = F.relu(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cell = nn.LSTMCell(64, 128)

    def forward(self, x):
        outputs = []
        h, c = None, None
        for i in range(x.shape[1]):
            tick()
            if h is None:
                h, c = self.cell(x[:, i, :])
            else:
                h, c = self.cell(x[:, i, :], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)


def main():
    # model = Model()
    # model = tvmodels.squeezenet1_0()
    model = LSTM()
    # model = nn.Transformer(d_model=64)
    print(model)
    start(model)
    model(torch.rand(1, 3, 64))
    # model(torch.rand(1, 10, 64), torch.rand(1, 10, 64))
    # model(torch.rand((1, 3, 224, 224)))
    # model(torch.rand(1, 128))
    graph = finish()
    # with connect():
    #     get_logger('main').info(graph)
    print(str(vz.assemble(graph)).replace("None", "null").replace("False", "false").replace("True", "true"))


if __name__ == "__main__":
    main()
