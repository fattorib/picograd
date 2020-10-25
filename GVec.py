import numpy as np

from Node import *

class GVec():
    
    def __init__(self,data,grad):
        self.value = data
        self.grad = grad
        self.graph = []
        
    def backward(self):
        gradient = 1
        for i in self.graph[::-1]:
            gradient *= i.deriv
        return gradient
    
    def print_graph(self):
        #Print computational graph
        graph_str = ''
        
        for i in self.graph:
            graph_str += i.operation + ' => '
            
        return graph_str[:-4]


#Scalar functions
def add_scalar_GVec(x,c):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'addition',[],1,1)
        x.value += c
        x.graph.append(node)
        
    else:
        node = Node(x.value,'addition',x.graph[-1],1,1)
        x.value += c
        x.graph.append(node)

    return x


def prod_scalar_GVec(x,c):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'product',[],c,c)
        x.value *= c
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'product',x.graph[-1],c,c)
        x.value *= c
        x.graph.append(node)
        
    return x
    
#Vector functions   
def add_GVec(x,y):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'addition',[],1,1)
        x.value += y.value
        x.graph.append(node)
        
    else:
        node = Node(x.value,'addition',x.graph[-1],1,1)
        x.value += y.value
        x.graph.append(node)
    
    #New node code here
    if y.graph == []:
        node = Node(y.value,'addition',[],1,1)
        y.value += x.value
        y.graph.append(node)
        
    else:
        node = Node(y.value,'addition',y.graph[-1],1,1)
        y.value += x.value
        y.graph.append(node)
    return x

def prod_GVec(x,y):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'product',[],y.value,y.value)
        x.value *= y.value
        x.graph.append(node)
        
    else:
        node = Node(x.value,'product',x.graph[-1],y.value,y.value)
        x.value *= y.value
        x.graph.append(node)
    
        #New node code here
    if y.graph == []:
        node = Node(y.value,'product',[],x.value,x.value)
        y.value *= x.value
        y.graph.append(node)
        
    else:
        node = Node(y.value,'product',y.graph[-1],x.value,x.value)
        y.value *= x.value
        y.graph.append(node)
    return x


def pow_GVec(x,c):

    #New node code here
    if x.graph == []:
        node = Node(x.value,'power',[],c*(x.value**(c-1)))
        x.value = x.value**c
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'power',x.graph[-1],c*(x.value**(c-1)))
        x.value = x.value**c
        x.graph.append(node)
        
    return x

def neg_GVec(x):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'negation',[],-1)
        x.value = -1*(x.value)
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'negation',x.graph[-1],-1)
        x.value = -1*(x.value)
        x.graph.append(node)
        
    return x

def exp_GVec(x):

    #New node code here
    if x.graph == []:
        node = Node(x.value,'exponentiation',[],np.exp(x.value))
        x.value = np.exp(x.value)
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'exponentiation',x.graph[-1],np.exp(x.value))
        x.value = np.exp(x.value)
        x.graph.append(node)
        
    return x

def recip_GVec(x):
    
    #New node code here
    if x.graph == []:
        node = Node(x.value,'reciprocal',[],(-1)/(x.value**2))
        x.value = 1/(x.value)
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'reciprocal',x.graph[-1],(-1)/(x.value**2))
        x.value = 1/(x.value)
        x.graph.append(node)
        
        
    return x

def root_GVec(x,n):
    #New node code here
    if x.graph == []:
        node = Node(x.value,'nth root',[],(1/n)*x**((-1/n)-1))
        x.value = x**(1/n)
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'nth root',x.graph[-1],(1/n)*x**((-1/n)-1))
        x.value = x**(1/n)
        x.graph.append(node)
        
        
    return x



if __name__ == "__main__":
    
    #Initialize 'vector'
    x = GVec(1,True)
    y = GVec(1,True)
    
    #Test function
    def f(x,y):
        return add_GVec(pow_GVec(x,2),pow_GVec(y,2))
    
    
    z = f(x,y)
    print(x.backward(),y.backward())
    

"""
To ADD:
    Efficient code for creating an array of values, 
    x = GVec_array([1,2,3])
    x.backward(), etc

"""
    
    
    
    
    
    
    
    



















