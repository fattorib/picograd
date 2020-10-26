#Operations consistent with numpy
import numpy as np

from Node import *

import Diff_array

"""
Following pseudocode of:
    1) Unbox to numpy array
    2) perform numpy operation
    3) Repackage as new node


"""


def add(x,y):
    
    if type(x) == Diff_array.array and type(y) == Diff_array.array:
        #Unpack values to numpy arrays
        x_val = x.value
        y_val = y.value
        
        #Perform numpy operation
        val = x_val+y_val
        
        #Update values and create nodes. Is this correct????
        x.value = val
        y.value = val
        
        if x.graph == []:
            x.graph.append(Node(x_val,None,'addition'))
        else:
            x.graph.append(Node(x_val,x.graph[-1],'addition'))
        
        if y.graph == []:
            y.graph.append(Node(y_val,None,'addition'))
        else:
            y.graph.append(Node(y_val,y.graph[-1],'addition'))
        
        return x
    
    
    elif type(x) != Diff_array.array:
        y_val = y.value
        #Perform numpy operation
        val = x + y_val
        
        y.value = val

        if y.graph == []:
            y.graph.append(Node(y.value,None,'scalar_addition'))
            
        else:
            y.graph.append(Node(y.value,y.graph[-1],'scalar_addition'))

                               
        return y
    
    elif type(y) != Diff_array.array:
        x_val = x.value
        val = y + x_val        
        #Update values and create nodes. Is this correct????
        x.value = val
               
        if x.graph == []:
            x.graph.append(Node(x.value,None,'scalar_addition'))
        
        else:
            x.graph.append(Node(x.value,x.graph[-1],'scalar_addition'))

        
        return x
        
def negative(x):
    #Unpack to numpy array
    x_val = x.value
    #Perform numpy operation
    x_val = -1*x.value
    
    #Update and repack
    x.value = x_val
        
    if x.graph == []:
        x.graph.append(Node(x_val,None,'negation'))     
    else:
        x.graph.append(Node(x_val,x.graph[-1],'negation'))

    return x


       
def exp(x):
    #Unpack to numpy array
    x_val = x.value
    
    x_val = np.exp(x.value)
    
    #Update and repack
    x.value = x_val

    if x.graph == []:
        x.graph.append(Node(x_val,None,'exponentiation'))
       
    else:
        x.graph.append(Node(x_val,x.graph[-1],'exponentiation'))

        
    return x
       
def reciprocal(x):
    #Unpack to numpy array
    x_val = x.value
    x_val = 1/(x.value)
    
    #Update and repack
    x.value = x_val
    #Perform numpy operation
    #NEED TO ADD ERROR HANDLING HERE    
    if x.graph == []:
        x.graph.append(Node(x.value,None,'reciprocal'))

    else:
        x.graph.append(Node(x.value,x.graph[-1],'reciprocal'))

    return x


def scale(x,c):
    #Unpack to numpy array
    x_val = x.value
    x_val = c*(x.value)
    
    #Update and repack
    x.value = x_val
    #Perform numpy operation
    if x.graph == []:
        x.graph.append(Node(x.value,None,'multiplication',c))

    else:
        x.graph.append(Node(x.value,x.graph[-1],'multiplication',c))

    return x

def power(x,c):
    #Unpack to numpy array
    x_val = x.value
    x_val = (x.value)**c
    
    #Update and repack
    x.value = x_val
    #Perform numpy operation  
    if x.graph == []:
        x.graph.append(Node(x.value,None,'power',c))

    else:
        x.graph.append(Node(x.value,x.graph[-1],'power',c))

    return x



if __name__ == "__main__":
    
    x = Diff_array.Array([1],True)
    
    y = Diff_array.Array([2,2,2],True)
    
    def logistic(x):
        return reciprocal(add(1,exp(negative(x))))
    
    z = logistic(x)
    
    
    def exponential(x):
        return exp(negative(x))
    
    # z = exponential(x)
    
    print(z.value[0])
    print(x.backward()[0])
    print(z.print_graph())
    
    
    
    
    
    
    
    