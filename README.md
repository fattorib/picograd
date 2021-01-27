# Picograd - Fully connected neural networks in Python 

<p align="center">
<img src="https://github.com/fattorib/picograd/blob/picograd/pics/autodiff.png" width="500">
</p>

## What is this?
In the process of learning ML and PyTorch, I decided to try and write my own neural network package in Python. It has been (relatively) succesful. 

Optimizing neural networks involves minimization of a cost function (the loss) from a very high dimensional space to <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{R}" title="\mathbb{R}" /></a>. Because of this, it would not be practical to compute hunderds of thousands of numerical derivatives, each requiring a separate forward pass of the network just for a single optimization step. Instead, we can use something called [Reverse Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation). You can think of it as repeatedly applying the chain to some very large compositions of functions. 


## How does it work?
Since what we are doing is computing the chain rule property at a bunch of nodes, we need a way to track how each node is transformed when an operation is applied to it. 
The picograd Tensor class is a wrapper around a numpy array with the following attributes and methods:
- `value`: This is the value of the node, stored as a NumPy array
- `parents`: A set containing all the parents of 
- `grad`: The gradient at this node, initialzed to a NumPy array of zeros in the shape of `value`
- `_backward`: The basic operation at this node
- `backward`: A method which computes the backward pass from this node. This means we are computing all gradients with respect to this node

For example, lets have two values `x=1` and `y=2`. If we add them together, we get `z=x+y`. In picograd, a new Tensor object for `z` is created, it has the following properties:
- `value`: `3`
- `parents`: `set(x,y)`
- `grad`: `0`
- `_backward`: A lambda function telling us how to update the gradients for `x` and `y`.

If we call `z.backward()`, the backward method is called to compute all gradients. But what about the gradient of `z`? For backpropogation, the node from which backward() is called is set to have a gradient of `1`. Note that this means we can only use this for scalar valued functions. 

Calling `z.backward()` builds the computational graph of all parent nodes starting at `z`. From this graph, we go through it in reversed topologically sorted order and apply the `_backward()` function for each node. 

## What actually works?
Any operation we would like to apply needs to have a corresponding backward pass method implemented. When dealing with functions operating on vectors and not just scalars, we implement these in the form of a Vector Jacobian Product or VJP. VJPs give us a more compact way to represent the gradient updates for a node. (EXPAND)

Operations for neural networks can be broken down into the following categories:
- Elementwise operations: `ReLU`,`Sigmoid`,`exp`, `softmax`
- Non-broadcasted binary operations: `dot`, `matmul`
- Broadcasted binary operations: `+,-,*`
- Reduction operations: `mean`, `sum`, `max`

Using the above categories, I have implemented the following in picograd:
- `ReLU,LeakyReLU,Sigmoid, Tanh, Softmax, LogSoftmax, Dropout, Log, Exp`
- `Dot, Matmul`
- `Add,Sub,Pow`, (elementwise) `Mul`, `Div`
- `mean, sum, max`

With these operations, you can construct all the pieces required to create a fully connected neural network. Add in an optimizer (SGD and Adam implemented) and you train the network! See `examples/train_MNIST.ipynb` for a neural network trained on MNIST. To run tests, see `test/test_tensor.py`.
### A note on broadcasting operations
The backward pass for broadcasted operations are a bit subtle. Say that we compute a linear pass on a set of inputs with a batch size greater than 1: output = Wx + b. In this computation, the bias vector is broadcasted to match the batch size dimension. If we naively compute the backward gradient now, our gradient will have the wrong size! Instead, what we need to do is explicitly compute the backward pass for 'broadcasting'. I found a very helpful overview of it [here](http://coldattic.info/post/116/). To summarize it, we define an operation, F, which represents the broadcasting explicitly. This would make our above linear pass: output = Wx+F(b). We can therefore compute the VJP for this (it corresponds to summation along the batch axes).

## Example

```python
from picograd.Tensor import Tensor

x = Tensor.eye(3)
y = Tensor([[2.0,0,-2.0]])
z = y.dot(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```


## How would you improve on this?
You can do a fair amount with just the operations I have implemented. There are a few different directions this project can take:

### 1. Adding more operations 
If you wanted to train any sort of [vision model](https://arxiv.org/abs/1409.1556), you would need to implement a 2d convolutional operation, as well as average and max pooling operations. If you wanted to train a [Transformer](https://arxiv.org/abs/1706.03762), you would need to implement a LayerNorm and a Concat operation. 

Aside from this, adding more loss functions could be helpful. Currently only Negative Log Likelihood loss and Mean-Squared Error loss are implemented. 

### 2. Adding an accelerator
In its current state, picograd only works on CPU. Training the MNIST model in the examples takes around 10-15 minutes. Training the exact same model on a GPU in PyTorch takes around 30 seconds. Yikes! Given that I don't have any background in C/C++, which is required for adding OpenCL or CUDA support, this will be a difficult task.

### 3. Code Reformatting/MiniTorch
As I was writing this up, I came across [MiniTorch](https://minitorch.github.io/), which describes itself as, "...a pure Python re-implementation of the Torch API designed to be simple, easy-to-read, tested, and incremental." In the future I plan on working through this and reformatting picograd to use the same structure as PyTorch. 

### References:
[tinygrad](https://github.com/geohot/tinygrad) and [micrograd](https://github.com/karpathy/micrograd) are two other implementations of neural networks in Python. They were very useful to have as references when working on this project.
