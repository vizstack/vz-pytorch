# vz-pytorch
Computation graph library for Pytorch models, built with Vizstack.

## Usage
This library allows you to produce a computation graph for Pytorch models that use autograd.

1. Import the necessary functions: `from vz_pytorch import start, finish`.
2. Begin recording the graph: `start(model)`.
2. Execute your model: `model(inputs)`.
3. Finish recording: `graph = finish()`

The `graph` object has `__view__()` defined to produce the computation graph using Vizstack.
