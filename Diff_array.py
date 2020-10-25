import numpy as np

from Node import *
from VJP_dict import dict_functions

class array():
    
    def __init__(self,data,grad):
        self.value = np.array([i for i in data])
        self.grad_len = len(data)
        self.grad_bool = grad
        self.grad = np.zeros(self.grad_len)
        self.graph =[Node(self.value,None,'root')]
        self.dtype = self.value.dtype
      
    
    def backward(self):
        grad = 1
                
        #Only works if we have single child for each node
        for i in self.graph[::-1]:

            grad = dict_functions[i.fun](grad,i.value,i.other)
            
        
        return grad
    
    def print_graph(self):
        return [(self.graph[i].fun,self.graph[i].value[0]) for i in range(0,len(self.graph))]
