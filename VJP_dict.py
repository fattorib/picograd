
import numpy as np
# Single functions
"""
val represents the value at the node

I'm not sure what args represents yet but I know that we need it lol

"""
dict_functions = {'Addition': lambda val, args: 1,
                  'Multiplication': lambda val, args: args,
                  'Subtraction': lambda val, args: -1,
                  'Negative': lambda val, args: -1,
                  'Sine': lambda val, args: np.cos(val),
                  'Cosine': lambda val, args: -np.sin(val),
                  'Natural Logarithm': lambda val, args: 1/val,
                  'Exponential': lambda val, args: val}
