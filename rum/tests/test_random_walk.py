import torch
from torch_geometric.data import Data
from rum.random_walk import uniform_random_walk


def test_shape():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    data = Data(edge_index=edge_index, num_nodes=6)
    walks, eids = uniform_random_walk(data, 2, 3)
    assert walks.shape == (2, 6, 3)
    assert eids.shape == (2, 6, 2)
