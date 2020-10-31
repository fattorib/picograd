# AutoDiff
Toy autodiff library for differentiating vector to scalar functions. 
## Working
- Gradient computation of basic functions
- Computational graph 

## Example 
Computing derivative of the logistic function at x = 0:
```
import Diff_array as d
import operations as ops

x = d.array([0],True)

def logistic(x):
        return ops.reciprocal(ops.add(1,ops.exp(ops.negative(x))))
        
z = logistic(x) 

print(x.backward())
0.25
```
