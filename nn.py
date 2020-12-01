import numpy as np
from Tensor import Tensor


class Linear():

    def __init__(self, in_feats, out_feats, bias=True):
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weights = Tensor.random_uniform(self.in_feats, self.out_feats)
        self.bias_flag = bias

        if self.bias_flag:
            self.bias = Tensor.random_uniform(1, self.out_feats)

    def forward(self, input):
        return input.dot(self.weights)+self.bias

    def __call__(self, input):
        return input.dot(self.weights)+self.bias


if __name__ == "__main__":

    in_feats = 4
    out_feats = 2

    linear_layer = Linear(in_feats, out_feats)

    input_tensor = Tensor([[3, -0.5, 2, 7], [3, -0.5, 2, 7]])

    # pred_tensor = Tensor([[2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8]])

    output_tensor = linear_layer(input_tensor)
    print(output_tensor)
