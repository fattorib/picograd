from Node import *
from GVec import *

import numpy as np


class GVec_array():
    
    def __init__(self,data,grad):
        self.data = np.array([i for i in data])
        self.grad_len = len(data)
        self.values = np.array([GVec(i,True) for i in self.data])
        self.grad_bool = grad
        self.grad = np.zeros(self.grad_len)
        
        
        
    def backward(self):
        for i in range(0,self.grad_len):
            gradient = self.values[i].backward()
            self.grad[i] = gradient
            
        return self.grad
    
    # def print_graph(self):
    #     #Print computational graph
    #     graph_str = ''
        
    #     for i in self.graph:
    #         graph_str += i.operation + ' => '
            
    #     return graph_str[:-4]


#Vector -> Scalar
def add_GVec(arr):
    
    total_sum = np.sum([x.value for x in arr])
    
    for i in range(0,len(arr)):
        q = arr[i]
        if q.graph == []:
            node = Node(q.value,'addition',[],1,1)
            q.value = total_sum
            q.graph.append(node)
            
        else:
            node = Node(q.value,'addition',q.graph[-1],1,1)
            q.value = total_sum
            q.graph.append(node)
            
    return q



#Vector -> Vector 

#Power
def array_GVec_pow(x,c):
    for e_i in x.values:
        pow_GVec(e_i,c)
    return x

#Hadamard Product
def array_GVec_Hadamard(x,y):
    
    for e_i,f_i in zip(x.values,y.values):
        prod_GVec(e_i,f_i)
        
    return x

def array_GVex_add(x,y):
    for e_i,f_i in zip(x.values,y.values):
        add_GVec(e_i,f_i)
        
    return x



        
def array_GVec_sum(x):
    #Sum elements in vector
    array_vals = x.values
    return add_GVec(array_vals)
    



if __name__ == "__main__":
    
    x = GVec_array([1,1],True)
    
    def f(z):
        return array_GVec_sum(array_GVec_pow(z,2))
    
    z = f(x)
    
    print(z.value)
    
    #Graph not updating
    print(x.backward())
    


    
    