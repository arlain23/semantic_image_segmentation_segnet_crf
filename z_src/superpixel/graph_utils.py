import torch
from torch_geometric.data import Data
import numpy as np
import z_src.utils.config as cfg


def generate_graph(unary_potentials, image):
    unary_potentials = np.swapaxes(unary_potentials, 0, 1)
    print("unary shape", unary_potentials.shape)
    h_len = cfg.IMAGE_SIZE['H']
    w_len = cfg.IMAGE_SIZE['W']

    num_edges = ((w_len - 1) * h_len * 2) + ((h_len - 1) * w_len * 2)
    edge_index = np.zeros((2, num_edges), dtype=int)

    # number of features is i, y, R, G, B and 21 unary features
    x = np.zeros((h_len * w_len, cfg.NUMBER_OF_CLASSES + 5), dtype=float)

    edge = 0
    for h in range(h_len):
        for w in range(w_len):
            # get edges

            index = h * h_len + w
            # x connections
            if w != w_len - 1:
                source = index
                target = index + 1

                edge_index[0][edge] = source
                edge_index[1][edge] = target
                edge += 1

                edge_index[0][edge] = target
                edge_index[1][edge] = source
                edge += 1

            # y connections
            if h != h_len - 1:
                source = index
                target = (h + 1) * h_len + w

                edge_index[0][edge] = source
                edge_index[1][edge] = target
                edge += 1

                edge_index[0][edge] = target
                edge_index[1][edge] = source
                edge += 1

            # features
            # unary
            x[index][0] = w
            x[index][1] = h
            x[index][2] = image[h][w][0]
            x[index][3] = image[h][w][1]
            x[index][4] = image[h][w][2]
            x[index][5:(5+cfg.NUMBER_OF_CLASSES)] = unary_potentials[index][:]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index)
    return graph
