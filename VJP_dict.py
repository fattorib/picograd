import numpy as np
"""
val represents the value at the node
parent_val is the value at the parent node
args represents any other arguments required for computation
"""
dict_functions = {'Addition': lambda val, parent_val, args: 1,
                  'Reciprocal': lambda val, parent_val, args: -1/(val)**2,
                  'Power': lambda val, parent_val, args: (args[0]*(val)**(args[0]-1)),
                  'Scaling': lambda val, parent_val, args: args[0],
                  'Shifting': lambda val, parent_val, args: 1,
                  'Multiplication': lambda val, parent_val, args: parent_val,
                  'Negative': lambda val, parent_val, args: -1,
                  'Sine': lambda val, parent_val, args: np.cos(val),
                  'Cosine': lambda val, parent_val, args: -np.sin(val),
                  'Natural Logarithm': lambda val, parent_val, args: 1/val,
                  'Exponential': lambda val, parent_val, args: np.exp(val)}
