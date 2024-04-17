import torch
import torch.nn as nn
import torch.nn.functional as F

class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)
        output = self.bn1(output)
        output = self.relu(output)
        #output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output

class TDM(nn.Module):

    def __init__(self, resnet):

        super().__init__()

        self.resnet = resnet
        if self.resnet:
            self.in_c = 640
        else:
            self.in_c = 64

        self.prt_self = SandGlassBlock(self.in_c)
        self.prt_other = SandGlassBlock(self.in_c)
     

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


    def dist(self,input,normalize=True):
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

    def weight(self,support):
        way, c, m = support.shape
        
        dist_prt_self, dist_prt_other = self.dist(support,normalize=True)
        

        dist_prt_self = dist_prt_self.view(-1, c) #5,64
        dist_prt_other, _ = dist_prt_other.min(dim=-2)
        dist_prt_other = dist_prt_other.view(-1, c) #5,64
     

        weight_prt_self = self.prt_self(dist_prt_self)
        #weight_prt_self = weight_prt_self.view(way, 1, c)
        weight_prt_other = self.prt_other(dist_prt_other)
        #weight_prt_other = weight_prt_other.view(way, 1, c)

        alpha_prt = 0.5
        #alpha_prt_qry = 0.5

        beta_prt = 1. - alpha_prt
        #beta_prt_qry = 1. - alpha_prt_qry

        weight_prt = alpha_prt * weight_prt_self + beta_prt * weight_prt_other
        #weight = alpha_prt_qry * weight_prt + beta_prt_qry * weight_qry_self

        return weight_prt

    def forward(self, support):
        weight = self.weight(support)
        #weight = self.add_noise(weight)

        return weight


    
