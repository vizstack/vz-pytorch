import inspect

from typing import List, Optional, Tuple, Any, Union, Dict
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn

import vizstack as vz

from vz_pytorch.special_functions import FUNCTIONS as special_functions

__all__ = ["start", "stop", "tracking", "tick", "name", "tag", "tag_image", "pause"]


class ComputationGraphNode:
    def __init__(self, payload: Any, variant: str):
        self._payload = payload
        assert variant in ["call", "data"]
        self._variant = variant
        if self._variant == "call":
            assert isinstance(payload, (nn.Module, FunctionContext))

        self.parent = None
        self.ancestors = []
        self.temporal_groups = [[]]

        self.edges: List[Tuple[str, ComputationGraphNode, str]] = []
        self.edges_to: List[Tuple[ComputationGraphNode, str, str]] = []

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

        self._view = None
        self._label_args = None
        self._tags = []
        # Create the fragment assembler immediately in case it mutates later
        if isinstance(self._payload, nn.Module):
            self._label_args = {"text": self._payload.__class__.__name__, "color": "purple"}
        elif isinstance(self._payload, torch.Tensor):
            try:
                # Check if this is a parameter or directly derived from a parameter
                if isinstance(self._payload, nn.Parameter) or (
                    len(self._payload.grad_fn.next_functions) == 1
                    and "AccumulateGrad" in str(self._payload.grad_fn.next_functions[0][0].name)
                ):
                    self._label_args = {
                        "text": f"Parameter{list(self._payload.shape)}",
                        "color": "green",
                    }
            except AttributeError:
                pass
            if self._label_args is None:
                self._label_args = {
                    "text": f"Tensor{list(self._payload.shape)}",
                    "color": "blue",
                }
        elif isinstance(self._payload, (float, int)):
            self._label_args = {
                "text": str(self._payload),
                "color": "orange",
            }
        else:
            self._view = vz.view(self._payload)

        # If the data has been relabeled, apply the label
        try:
            self.name(self._payload.cg_name)
        except AttributeError:
            pass

        # If any tags have been added to the data, apply those tags
        try:
            for tag, kind in self._payload.cg_tags:
                self.tag(tag, kind)
        except AttributeError:
            pass

    def child(self, node: "ComputationGraphNode"):
        node.ancestors = self.ancestors + [self]
        node.parent = self
        self.temporal_groups[-1].append(node)
        return node

    def edge(
        self, start_port: str, end: "ComputationGraphNode", end_port: str,
    ):
        self.edges.append((start_port, end, end_port))
        end.edges_to.append((self, start_port, end_port))

    def remove_edge(self, start_port: str, end: "ComputationGraphNode", end_port: str):
        self.edges.remove((start_port, end, end_port))
        end.edges_to.remove((self, start_port, end_port))

    def name(self, name):
        assert self._variant == "data"
        if self._label_args is not None:
            self._label_args["text"] = f"{name}"

    def tag(self, tag, kind):
        if kind == "image":
            self._tags.append(vz.Image(tag))
        elif kind == "text":
            self._tags.append(
                vz.Token(str(tag), color=self._label_args["color"] if self._label_args is not None else "blue")
            )
        else:
            raise ValueError(f"Unsupported tag of kind {kind}.")

    def replace_with_edges(self) -> bool:
        """
        Checks if this is a data node which only exists to connect to other data nodes at higher container levels. If
        so, reroute all edges through this node to circumvent it; this node then no longer needs to be added to the
        graph.
        """
        if self._variant != "data" or len(self.edges_to) == 0:
            return False
        assert len(self.edges_to) == 1
        edges = [(edge, edge) for edge in self.edges]
        connects_to_call_input = False
        connected_data = []
        while len(edges) > 0:
            (start_port, end_node, end_port), first_edge = edges.pop()
            if end_node._variant == "data":
                connected_data.append((end_node, first_edge))
            elif "i" in end_port:
                connects_to_call_input = True
                break
            else:
                edges.extend([(edge, first_edge) for edge in end_node.edges if edge[0] == end_port])
        if not connects_to_call_input and len(connected_data) > 0:
            assert len(self.edges_to) == 1
            # remove the edge from the preceding node to self
            creator_node, creator_port, input_port = self.edges_to[0]
            creator_node.remove_edge(creator_port, self, input_port)
            assert len(self.edges_to) == 0
            for end_node, first_edge in connected_data:
                # create an edge from the preceding node to the end of the self-starting edge
                creator_node.edge(creator_port, first_edge[1], first_edge[2])
            return True
        return False

    def has_ancestor(self, ancestor: "ComputationGraphNode"):
        return ancestor in self.ancestors

    def is_expanded(self):
        return not isinstance(self._payload, nn.Module) or "torch.nn" not in self._payload.__module__

    def create_temporal(self):
        if len(self.temporal_groups[-1]) > 0:
            self.temporal_groups.append([])

    def __view__(self):
        # TODO We can't do this on initialization, since it currently causes an infinite loop of tracked functions
        if self._label_args is not None:
            self._view = vz.Token(**self._label_args)
            if len(self._tags) > 0:
                self._view = vz.Switch(["label"] + [str(i) for i in range(len(self._tags))], {"label": self._view})
                for i, tag in enumerate(self._tags):
                    self._view.item(str(i), tag)
        return self._view


class FunctionContext:
    def __init__(self, fn):
        self.fn = fn
        self.cg_input_locations: List[Tuple[ComputationGraphNode, str]] = []
        self.text = self.fn.__name__

    def __view__(self):
        return vz.Token(self.text, color="red")


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
_MAGIC_METHODS += ["__i" + fn_name.lstrip("__") for fn_name in _MAGIC_METHODS]


class Tracker:
    def __init__(self, models: Optional[List[nn.Module]] = None):
        self._models = models if models is not None else []
        self._hooks = []
        self._node: Optional[ComputationGraphNode] = None
        self._old_modules: Dict[str, Dict[str, Any]] = dict()
        self._paused = False

    def start(self) -> None:
        self._node = ComputationGraphNode(None, "data")
        self._old_modules["torch"] = dict()
        self._old_modules["torch.Tensor"] = dict()
        for var, value in vars(torch).items():
            if callable(value) and not inspect.isclass(value):
                setattr(torch, var, self._wrap_fn(value))
                assert getattr(torch, var).cg_is_wrapped
                self._old_modules["torch"][var] = value

        for var in dir(torch.Tensor):
            value = getattr(torch.Tensor, var)
            if callable(value) and var in _MAGIC_METHODS:
                setattr(torch.Tensor, var, self._wrap_fn(value))
                self._old_modules["torch.Tensor"][var] = value

        for model in self._models:
            for module in model.modules():
                self._hooks.append(module.register_forward_pre_hook(self._on_call))
                self._hooks.append(module.register_forward_hook(self._on_return))

    def stop(self):
        nodes = [(self._node, None)]
        self._node = None
        for var, value in self._old_modules["torch"].items():
            setattr(torch, var, value)
        for var, value in self._old_modules["torch.Tensor"].items():
            setattr(torch.Tensor, var, value)
        for hook in self._hooks:
            hook.remove()

        d = vz.Dag()
        node_to_id: Dict[ComputationGraphNode, str] = {}
        node_to_parent: Dict[ComputationGraphNode, str] = {}
        while len(nodes) > 0:
            node, parent = nodes.pop()
            if not node.replace_with_edges():
                node_to_id[node] = str(len(node_to_id))
                node_to_parent[node] = parent
                d.node(
                    node_to_id[node],
                    parent=parent,
                    item=node,
                    is_expanded=node.is_expanded(),
                    flow_direction="south" if len(node.temporal_groups) == 1 else "east",
                    label=node._label_args["text"] if node._label_args is not None else None,
                )
                if len(node.temporal_groups) > 1:
                    for t in range(len(node.temporal_groups)):
                        d.node(
                            f"{node_to_id[node]}_{t}",
                            parent=node_to_id[node],
                            is_interactive=False,
                            is_visible=False,
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
                # TODO: HACK HACK HACK
                temporal = (
                    node_to_parent[node] != node_to_parent[end_node]
                    and node_to_parent[node] != node_to_id[end_node]
                    and node_to_id[node] != node_to_parent[end_node]
                    and node.parent != end_node
                    and end_node.parent != node
                )
                d.edge(
                    {"id": node_to_id[node], "port": start_port,},
                    {"id": node_to_id[end_node], "port": end_port,},
                    temporal=temporal,
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
        if self._node is not None:
            self._node.create_temporal()

    def name(self, data, name):
        try:
            data.cg_name = name
            data.cg_location[0].name(name)
        except AttributeError:
            pass

    def tag(self, data, tag):
        try:
            if not hasattr(data, "cg_tags"):
                data.cg_tags = []
            data.cg_tags.append((tag, "text"))
            data.cg_location[0].tag(tag, "text")
        except AttributeError:
            pass

    def tag_image(self, data, image_str):
        try:
            if not hasattr(data, "cg_tags"):
                data.cg_tags = []
            data.cg_tags.append((image_str, "image"))
            data.cg_location[0].tag(image_str, "image")
        except AttributeError:
            pass

    @contextmanager
    def pause(self):
        self._paused = True
        yield
        self._paused = False

    def _on_call(self, fn: Union[nn.Module, FunctionContext], arguments: Tuple[Any, ...]):
        if self._paused:
            return
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
                    data_node = self._node.child(ComputationGraphNode(arg_value, "data"))
                    data_node.edge("o0", call_node, f"i{i}")
            argument_locations.append(arg_iterable_locations)
        fn.cg_input_locations = argument_locations
        self._node = call_node

    def _on_return(self, fn: Union[nn.Module, FunctionContext], arguments: Tuple[Any], outputs: Any):
        if self._paused:
            return
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
                try:
                    output_value.cg_location = (data_node, "o0")
                except AttributeError:
                    # The output is a primitive and cannot have a location
                    pass
        self._node = self._node.parent

    # TODO: show cracked list arguments in correct order with constraints somehow
    # TODO: make it so printing doesnt create hundreds of extra nodes
    # TODO: cleanup graph and node and separate out functions
    # TODO: primitive inputs (i.e., integers into __getitem__)
    # TODO: deal with in-place ops (maybe done?)
    # TODO: make modules return subclassed primitives so they can be used as inputs later

    def _wrap_fn(self, fn):
        def _wrapped(*args, **kwargs):
            if self._paused:
                return fn(*args, **kwargs)
            ctx = FunctionContext(fn)
            if fn.__name__ not in special_functions:
                inputs = args + tuple(kwargs.values())
            else:
                inputs, ctx.text = special_functions[fn.__name__](args, kwargs)
            self._on_call(ctx, inputs)
            outputs = fn(*args, **kwargs)
            is_output_iterable = isinstance(outputs, (list, tuple))
            if not is_output_iterable:
                outputs = (outputs,)
            if not is_output_iterable:
                outputs = outputs[0]
            self._on_return(ctx, inputs, outputs)
            return outputs

        _wrapped.cg_is_wrapped = True
        return _wrapped


_tracker = None


def start(model):
    global _tracker
    assert _tracker is None
    _tracker = Tracker(models=[model])
    _tracker.start()


def stop():
    global _tracker
    assert _tracker is not None
    output = _tracker.stop()
    _tracker = None
    return output


def tick():
    global _tracker
    if _tracker is not None:
        _tracker.tick()


def name(data, name):
    global _tracker
    if _tracker is not None:
        _tracker.name(data, name)


def tag(data, tag):
    global _tracker
    if _tracker is not None:
        _tracker.tag(data, tag)


def tag_image(data, image):
    global _tracker
    if _tracker is not None:
        _tracker.tag_image(data, image)


def tracking():
    global _tracker
    return _tracker is not None


@contextmanager
def pause():
    global _tracker
    if _tracker is not None:
        with _tracker.pause():
            yield
    else:
        yield
