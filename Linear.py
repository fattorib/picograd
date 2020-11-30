from Tensor import Tensor
from TensorOps import linear_mm, sum


class Linear():
    # Holds the linear layers
    def __init__(self, in_feats, out_feats, bias=False):
        # MISSING BIAS!!!

        self.weights = Tensor.random(out_feats, in_feats)

    def forward(self, input):
        return linear_mm(self.weights, input)

    def __call__(self, input):
        return Linear.forward(self, input)


if __name__ == "__main__":
    out_feats = 10
    in_feats = 20
    batch_size = 12

    fc1 = Linear(in_feats, out_feats)
    input_vector = Tensor.ones((batch_size, in_feats))

    out = fc1(input_vector)
    print(out.shape)
    out = out.sum()
    print(out)
    # out.backward()
