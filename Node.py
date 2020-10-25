
class Node():    
    def __init__(self,value,operation,parent,deriv,*args):
        self.value = value
        self.operation = operation
        self.parent = parent
        self.deriv = deriv
        self.other = args
        
        