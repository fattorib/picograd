"""
Basic functions:
    1) Addition
    2) Multiplication
    3) exponentiation 
    4) Division
    5) Subtraction
    



"""


class Node():
    
    def __init__(self,value,operation,parent,deriv,*args):
        self.value = value
        self.operation = operation
        self.parent = parent
        self.deriv = deriv
        self.other = args
        
        