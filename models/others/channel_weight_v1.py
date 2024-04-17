import torch
import torch.nn as nn
import torch.nn.functional as F


def dist(input,normalize=True):
    way, c, m = input.shape
    input_C_gap = input.mean(dim=-2)

    input = input.reshape(way * c, m)
    input = input.unsqueeze(dim=1)
    input_C_gap = input_C_gap.unsqueeze(dim=0)

    dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
    if normalize:
        dist = dist / m
    dist = dist.reshape(way, c, -1)
    dist = dist.transpose(-1, -2)

    indices_way = torch.arange(way)
    indices_1 = indices_way.repeat_interleave((way - 1))
    indices_2 = []
    for i in indices_way:
        indices_2_temp = torch.cat((indices_way[:i], indices_way[i + 1:]),
                                   dim=-1)
        indices_2.append(indices_2_temp)
    indices_2 = torch.cat(indices_2, dim=0)

    dist_self = dist[indices_way, indices_way]
    dist_other = dist[indices_1, indices_2]
    dist_other = dist_other.view(way, way-1, -1)

    return dist_self, dist_other

def weight_channel(support):
    way, c, m = support.shape
    dist_prt_self, dist_prt_other=dist(support,normalize=True)
    dist_prt_self = dist_prt_self.view(-1, c) #5,64
    dist_prt_other, _ = dist_prt_other.min(dim=-2) 
    dist_prt_other = dist_prt_other.view(-1, c) #5,64
    dist_prt_self_max,_=dist_prt_self.max(dim=1,keepdim=True)
    dist_prt_self_max,_=dist_prt_other.max(dim=1,keepdim=True)
    weight_prt_self=dist_prt_self/dist_prt_self_max
    weight_prt_other=dist_prt_other/dist_prt_self_max
    weight=0.4*(1-weight_prt_self)+0.6*weight_prt_other
    return weight
