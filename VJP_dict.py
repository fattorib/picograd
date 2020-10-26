
import numpy as np
#Single functions

dict_functions= {'root': lambda grad,ans,x,args: grad,
                 'exponentiation': lambda grad,ans,x,args: ans*(grad),
                 'negation': lambda grad,ans,x,args: -grad,
                 'scalar_addition':lambda grad,ans,x,args: grad,
                 'reciprocal': lambda grad,ans,x,args: (-1)*grad*((1/x)**2),
                 'multiplication': lambda grad,ans,x,args: args[0]*grad,
                 'power': lambda grad,ans,x,args: (grad*args[0])*x**(args[0]-1)
                 }



