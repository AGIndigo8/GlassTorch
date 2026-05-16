import torch

def dynamic_index(T: torch.Tensor, targets: dict[int, int], default: int = 0):
    """
    T            : tensor of arbitrary order N
    targets      : {axis: scope} are the locations on the tensor that you want to index, framed as deviations from the 0-index. Axis is the direction, and scope is the deviation, e.g. {3: 2, 6: 4}
    default      : index to use on all unspecified axes
    
    Returns a scalar tensor (0-dim).
    """
    N = T.dim()
    idx = tuple(targets.get(ax, default) for ax in range(N))
    return T[idx]

def dynamic_set(T: torch.Tensor, targets: dict[int, int], value: torch.Tensor, default: int = 0):
    """
    T            : tensor of arbitrary order N
    targets      : {axis: scope} are the locations on the tensor that you want to index, framed as deviations from the 0-index. Axis is the direction, and scope is the deviation, e.g. {3: 2, 6: 4}
    value        : the value to set at the indexed location
    default      : index to use on all unspecified axes
    
    Sets the value at the specified location on the tensor.
    """
    N = T.dim()
    idx = tuple(targets.get(ax, default) for ax in range(N))
    T[idx] = value

def list_to_target_dict(axes: list[int], scope: int = 1):
    targets = {axis: scope for axis in axes}
    return targets

'''
This function is used to dynamically grab the aspect of the cubic tensor input, T, defined by the axes in the list where all other axes are determined at the default index.
For our purposes, the defult index will always be 0, as the other aspects are computationally irrelevant in the energy calculation (as per the whitepaper).

input
T: a tensor of arbitrary order N
axes: a list of axes to grab from the tensor, e.g. [0, 3, 6]
default: the index to use for all other axes, e.g. 0
'''
def dynamic_aspect(T: torch.Tensor, axes: list[int], default: int = 0):
    N = T.dim()
    idx = tuple(slice(None) if ax in axes else default for ax in range(N))
    return T[idx]