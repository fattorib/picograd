import numpy as np

from Node import *
from VJP_dict import dict_functions

import networkx as nx
import matplotlib.pyplot as plt


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
        for i in self.graph[::-1]:
            if i.parent is not None:                
                grad = dict_functions[i.fun](grad,i.value,i.parent.value,i.other)       
            else:
                grad = dict_functions[i.fun](grad,i.value,1,i.other)
                
        return grad[0]
    
    def print_graph(self):
        return [(self.graph[i].fun,self.graph[i].value[0]) for i in range(0,len(self.graph))]
    
    
    def plot_computational_graph(self):
        
        
        
        G = nx.Graph()
        
        node_list =[]
        
        for i in range(0,len(self.graph)):
            G.add_node(self.graph[i].fun)
            node_list.append(self.graph[i].fun)
        
    
        
        for i in range(0,len(node_list)-1):
            
            G.add_edge(node_list[i],node_list[i+1])
        
        nx.draw(G, with_labels=True, font_weight='bold')

        
        plt.show()
        return None
