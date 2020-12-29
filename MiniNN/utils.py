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


# ------------Pooling-------------

def pool_stack(filters, kernel_size, stride):
    # Take our filters and convert them into a 'stack' of kernel_size arrays
    # Useful as this allows us to get argmax and max for backprop in MaxPool2d
    return None


def pool_unstack(filter_stack, kernel_size, stride):
    # Use after pool_stack to recover original filter shape
    return None


def pad_filters(filter_stack):
    # Use for padding filters to ensure proper kernels. Using zero padding
    padded_arr = np.pad(filter_stack, (1, 1), 'constant',
                        constant_values=(0, 0))
    return padded_arr[1:-1]


if __name__ == "__main__":
    filter_stack = np.random.rand(1, 10, 10)
    # print(filter_stack)

    print(pad_filters(filter_stack))
