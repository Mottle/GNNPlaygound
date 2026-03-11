import torch
from torch_geometric.data import Data


def test_layer_forward():
    from rum.layers import RUMLayer

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    data = Data(edge_index=edge_index, num_nodes=6)
    layer = RUMLayer(
        in_features=16, out_features=8, original_features=16, num_samples=2, length=3
    )
    h = torch.ones(6, 16)
    y0 = torch.ones(6, 16)
    out, loss = layer(data, h, y0)
    assert out.shape[-1] == 8
