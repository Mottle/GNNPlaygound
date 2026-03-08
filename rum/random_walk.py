from torch_geometric.utils import random_walk
import torch
from functools import partial


def uniform_random_walk(data, num_samples, length, subsample=None):
    """
    Random walk on a graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph data.
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk.

    Returns
    -------
    walks : Tensor
        The random walks.
    """
    edge_index = data.edge_index
    if subsample is None:
        nodes = torch.arange(data.num_nodes)
        num_nodes = data.num_nodes
        nodes = nodes.repeat(num_samples)
    else:
        nodes = subsample.repeat(num_samples)
        num_nodes = subsample.size(0)
    walks = random_walk(edge_index, nodes, walk_length=length)
    walks = walks.view(num_samples, num_nodes, length)
    # PyG random_walk 不返回 eids，保持接口一致返回 None
    return walks, None


# @torch.jit.trace(example_inputs=(torch.zeros(10, 10, 10)))


def uniqueness(walk):
    """
    Compute the uniqueness of a random walk.

    Parameters
    ----------
    walk : Tensor
        The random walk.

    Returns
    -------
    uniqueness : Tensor
        The uniqueness of the random walk.
    """
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal
