
import numpy as np
#Single functions

dict_functions= {'root': lambda grad,ans,x: grad,
                 'exponentiation': lambda grad,ans,x: ans*(grad),
                 'negation': lambda grad,ans,x: -grad,
                 'scalar_addition':lambda grad,ans,x: grad,
                 'reciprocal': lambda grad,ans,x: (-1)*grad*(ans**2),
                 'multiplication': lambda grad,ans,x: x*grad
                 }



