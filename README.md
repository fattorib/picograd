# MiniNN

MiniNN is a (mini) neural network package. It is an Automatic Differentiation library with some basic neural network operations (see below for all supported)

# OpenCL Support
Time to learn OpenCL...
| Function          | CPU | GPU |
|-------------------|-----|-----|
| Linear            | Yes | No  |
| Dropout           | Yes | No  |
| LogSoftmax        | Yes | No  |
| Softmax           | Yes | No  |
| ReLU              | Yes | No  |
| Sigmoid           | Yes | No  |
| Tanh              | Yes | No  |
| Add               | Yes | No  |
| Mul               | Yes | No  |
| Reshape           | Yes | No  |
| Sum (reduce)      | Yes | No  |
| Mean (reduce)     | Yes | No  |
| Exp (elementwise) | Yes | No  |
| Log (elementwise) | Yes | No  |

# Implemented

## Layers
- Linear
- Dropout 
- LogSoftmax

## Activations
- ReLU
- Sigmoid
- Tanh

## Optimizers
- Stochastic Gradient Descent
- Adam

## Loss Functions
- CrossEntropyLoss
- MSELoss

# Examples
Adding a collection of examples. See MNIST example [here](Examples/train_MNIST.ipynb)

# Future Work
- Add convolutional layers (requires adding MaxPool2d, AvgPool2d, Conv2d)
- Add support for Gpus (CuPY looks like a good start)
- Train Cifar-10 model
- RNN of some sort 
