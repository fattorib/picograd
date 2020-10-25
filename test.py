

import Diff_array as d

import operations as ops


if __name__ == "__main__":
    
    x = d.array([1],True)
    
    def logistic(x):
        return ops.reciprocal(ops.add(1,ops.exp(ops.negative(x))))
    
    # z = logistic(x) 
    
    def exp2x(x):
        return ops.exp(ops.scale(ops.negative(x),2))
    
    z = exp2x(x)
    
    print(z.value[0])
    print(x.backward()[0])
    print(z.print_graph())