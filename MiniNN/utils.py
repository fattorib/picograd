# Utility functions
import numpy as np


# ----------Broadcasting------------

# Unbroadcasting is needed for backward passes on broadcasted tensors.
# Useful resource: http://coldattic.info/post/116/

def unbroadcast(out, in_sh):
    # Sum along the batch axis and then reshape after
    sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]
                      == 1 and out.shape[i] > 1]) if in_sh != (1,) else None
    return out.sum(axis=sum_axis).reshape(in_sh)
