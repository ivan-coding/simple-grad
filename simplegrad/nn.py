from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import pairwise

from simplegrad.engine import Number, Value

InitRange = tuple[float, float]


class Neuron:
    def __init__(self, input_size: int, init_range: InitRange):
        self.weights = [Value(random.uniform(*init_range)) for _ in range(input_size)]
        self.bias = Value(random.uniform(*init_range))

    def __call__(self, inputs: Sequence[Value | Number]) -> Value:
        weighted_sum = sum(
            (weight * x for weight, x in zip(self.weights, inputs, strict=True)),
            self.bias,
        )
        return weighted_sum.tanh()

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


class Layer:
    def __init__(self, input_size: int, output_size: int, init_range: InitRange):
        self.neurons = [Neuron(input_size, init_range) for _ in range(output_size)]

    def __call__(self, inputs: Sequence[Value | Number]) -> Value | list[Value]:
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(self, input_size: int, layer_sizes: Sequence[int], init_range: InitRange):
        sizes = [input_size, *layer_sizes]
        self.layers = [
            Layer(layer_input, layer_output, init_range)
            for layer_input, layer_output in pairwise(sizes)
        ]

    def __call__(self, inputs: Sequence[Value | Number]) -> Value | list[Value]:
        activations = inputs
        for layer in self.layers:
            activations = layer(activations)
        return activations

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = 0.0


def sum_squared_error(predictions: Sequence[Value], targets: Sequence[Number]) -> Value:
    return sum(
        (
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets, strict=True)
        ),
        Value(0.0),
    )
