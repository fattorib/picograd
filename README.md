# Picograd - Fully connected neural networks in Python 

## What is this?
In the process of learning ML and PyTorch, I decided to try and write my own neural network package in Python. It has been (relatively) succesful. 

Optimizing neural networks involves minimization of a cost function (the loss) from a very high dimensional space to <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{R}" title="\mathbb{R}" /></a>. Because of this, it would not be practical to compute hunderds of thousands of numerical derivatives, each requiring a separate forward pass of the network just for a single optimization step. Instead, we can use something called [Reverse Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation). Think of it as repeatedly applying the chain to some very large compositions of functions. (Add a bit more explanation here)


## How does it work?
Since what we are doing is computing the chain rule property at a bunch of nodes, we need a way to track how each node is transformed when an operation is applied to it. 
The picograd tensor is a wrapper around a numpy array with the following attributes and methods:
- `value`: This is the value of the node, stored as a NumPy array
- `parents`: A set containing all the parents of 
- `grad`: The gradient at this node, initialzed to a NumPy array of zeros in the shape of `value`
- `_backward`: The basic operation at this node
- `backward`: A method which computes the backward pass from this node. This means we are computing all gradients with respect to this node

For example, lets have two values $x$ and $y$. If we add them together, we get $z=x+y$. We say that the parents of $z$ are $x$ and $y$.  

## What works?

## How would you improve on this?
