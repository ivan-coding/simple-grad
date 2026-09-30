from __future__ import annotations

import math

Number = int | float


class Value:
    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        operation: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = children
        self._op = operation
        self._backward = lambda: None

    def __add__(self, other: Value | Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: Number) -> Value:
        if not isinstance(exponent, Number):
            raise TypeError("Only int and float exponents are supported")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        nodes_in_topological_order = []
        visited = set()

        def visit(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                nodes_in_topological_order.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(nodes_in_topological_order):
            node._backward()

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | Number) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | Number) -> Value:
        return self * other**-1

    def __radd__(self, other: Number) -> Value:
        return self + other

    def __rsub__(self, other: Number) -> Value:
        return other + (-self)

    def __rmul__(self, other: Number) -> Value:
        return self * other

    def __rtruediv__(self, other: Number) -> Value:
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
