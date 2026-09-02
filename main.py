# todo
# final goal:
# train ML that would distinguish different numbers based on pictures (on any ML of this kind)
#
# steps:
# build my own micrograd
# learn basics of torch
# switch to torch
# implement layers
# implement loss calculation
# train on simple data to predict simple formulars
# create ml model to predict something harder (the next stock price or number on image, ...)

from draw import draw_dot

class Value:
    def __init__(self, data: float, _children=(), op = "", label = ""):
        self.data = data
        self._prev = set(_children)
        self._op = op
        self.label = label
        self.grad = 0.0

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data)

    def __truediv__(self, other):
        return Value(self.data/other.data)



    def __str__(self):
        return f"Value: {self.data}"

if __name__ == '__main__':
    a = Value(1.0, label="a")
    b = Value(2.0, label="b")
    c = a + b
    print(c)
    res = draw_dot(c)

