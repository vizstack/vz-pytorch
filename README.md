# vz-pytorch
Computation graph library for Pytorch models, built with Vizstack.

## Usage
This library allows you to produce a computation graph for Pytorch models that use autograd.

1. Create a `Tracker` object.
2. Pass the `model` using `tracker.model(model)`.
3. Call `tracker.start()`
4. Call `model(inputs)` as much as they want; every call is tracked.
5. Call `graph = tracker.stop()` to build the computation graph object. This object has `__view__()` defined to produce the computation graph using Vizstack.
