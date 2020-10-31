
import numpy as np
# Single functions

dict_functions = {'Addition': lambda x, args: 1,

                  'Multiplication': lambda x, args: args,
                  'Subtraction': lambda x, args: -1,


                  'Sine': lambda x, args: np.cos(x),
                  'Cosine': lambda x, args: -np.sin(x),
                  'Natural Logarithm': lambda x, args: 1/x,
                  'Exponential': lambda x, args: x}
