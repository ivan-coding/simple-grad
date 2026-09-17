# todo
# final goal:
# train ML that would distinguish different numbers based on pictures (on any ML of this kind)
#
# steps:
# build my own micrograd
# implement loss function and parameters adjustment based on it
# learn basics of torch
# switch to torch
# implement layers
# implement loss calculation
# train on simple data to predict simple formulars
# create ml model to predict something harder (the next stock price or number on image, ...)
import math

from draw import draw_dot

class Value:
    def __init__(self, data: float, _children=(), op = "", label = ""):
        self.data = data
        self._prev = set(_children)
        self._op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None


    def __add__(self, other):
        def _back():
            self.grad += 1
            other.grad += 1
        res_v = Value(self.data + other.data, (self, other), "+")
        res_v._backward = _back
        return res_v

    def __mul__(self, other):
        def _back():
            self.grad += other.data
            other.grad += self.data
        res_v = Value(self.data * other.data, (self, other), "*")
        res_v._backward = _back
        return res_v

    def tanh (self):
        t = self.data
        new_data = (math.exp(2 * t) - 1) / (math.exp(2 * t) + 1)
        def _back():
            self.grad = 1 - new_data**2
        res_value = Value(new_data, (self,), "tanh")
        res_value._backward = _back
        return res_value


    def __truediv__(self, other):
        return Value(self.data/other.data)



    def __str__(self):
        return f"Value: {self.data}"


    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

if __name__ == '__main__':
    a = Value(0.4, label="a")
    b = Value(0.5, label="b")
    c = Value(-0.4, label="c")
    d = Value(0.2, label="d")
    r = a * b + a * c + d
    o = r.tanh()
    o.backward()
    res = draw_dot(o)

