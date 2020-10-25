import numpy as np


from autodiff import Node


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
    
    
def add_GVec(x,c):
    
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


def prod_GVec(x,c):
    
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


def pow_GVec(x,c):

    #New node code here
    if x.graph == []:
        node = Node(x.value,'power',[],c*(x.value**(c-1)))
        x.value = np.exp(x.value)
        x.graph.append(node)
        
        
    else:
        node = Node(x.value,'power',x.graph[-1],c*(x.value**(c-1)))
        x.value = np.exp(x.value)
        x.graph.append(node)
        
    return x


if __name__ == "__main__":
    
    #Initialize 'vector'
    x = GVec(0,True)
    
    #Test function
    def logistic(z):
        return recip_GVec(add_GVec(exp_GVec(neg_GVec(z)),1))
    
    def linear(z,w,b):
        
        return add_GVec(prod_GVec(z,w),b)
    
    def cubic(x):
        return pow_GVec(x,3)
    
    x = logistic(x)
    
    grad = x.backward()
    
    print(grad)
    print(x.print_graph())
    
    
    
    
    
    
    



















