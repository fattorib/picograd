
import Diff_array as d

import operations as ops

if __name__ == "__main__":
    
    x = d.array([1],True)
    
    def logistic(x):
        return ops.reciprocal(ops.add(1,ops.exp(ops.negative(x))))
    
    def exp2x(x):
        return ops.exp(ops.scale(ops.negative(x),2))
      
    def cubic(x):
        return ops.power(ops.add(x,1),3)
        
    z = logistic(x) 
    # z = exp2x(x)
    # z = cubic(x)
    
    
    print(z.value[0])
    print(x.backward()[0])
    print(z.print_graph())
    # z.plot_computational_graph()
   