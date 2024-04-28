import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import Conv_4
from torchvision.utils import save_image
import torchvision.utils as vutils
import os
class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25 

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(9),requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   
    

    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        d=feature_map.size(1)
        h=feature_map.size(2)
        w=feature_map.size(3)
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        feature_map_d=feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous()

        feature_map_h=feature_map.permute(0,1,3,2).contiguous().view(batch_size,-1,h)
        feature_map_w=feature_map.view(batch_size,-1,w).contiguous()
        return feature_map_d,feature_map_h,feature_map_w
    

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=False):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not
        eps=1e-10
        #query=query/(query.norm(dim=1).unsqueeze(1)+eps)
        support=support/(support.norm(dim=2).unsqueeze(2)+eps)
        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution
        
        if Woodbury:
            # correspond to Equation 10 in the paper
            
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        #dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        
        return Q_bar

    
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
 
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        alpha1 = self.r[2]
        beta1 = self.r[3]
        alpha2 = self.r[4]
        beta2 = self.r[5]
        weight_d=self.r[6].exp()+1e-6
        weight_h=self.r[7].exp()+1e-6
        weight_w=self.r[8].exp()+1e-6
        #rho = beta.exp()
        feature_map_d,feature_map_h,feature_map_w = self.get_feature_map(inp)#inp:75*3*84*84 输出：75*25*64,75*320*5
        
        #d reconsturct
        support = feature_map_d[:way*shot].view(way, shot*resolution , d) #5*125*64
        query = feature_map_d[way*shot:].view(way*query_shot*resolution, d) #1250*64
        
        Q_bar_d = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        dist_d=(Q_bar_d-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way


        #h reconsruct
        support1 = feature_map_h[:way*shot].view(way,-1,5) #5*125*64
        query1 = feature_map_h[way*shot:].view(-1, 5) #1250*64
        
        Q_bar_h = self.get_recon_dist(query=query1,support=support1,alpha=alpha1,beta=beta1) # # way, way*query_shot*r, h
        #对Q_bar_h作reshape
        Q_bar_h=Q_bar_h.view(way,way*query_shot,d,-1).permute(0,1,3,2).contiguous()# way, way*query_shot,r,d
        Q_bar_h=Q_bar_h.view(way,-1,d).contiguous() # way, way*query_shot*r, d
        dist_h=(Q_bar_h-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way

        ######
        #w reconstruct
        support2 = feature_map_w[:way*shot].view(way,-1,5) #5*125*64
        query2 = feature_map_w[way*shot:].view(-1, 5) #1250*64
        
        Q_bar_w = self.get_recon_dist(query=query2,support=support2,alpha=alpha2,beta=beta2) # # way, way*query_shot*r, h

        #对Q_bar_h作reshape
        Q_bar_w=Q_bar_w.view(way,way*query_shot,d,-1).permute(0,1,3,2).contiguous()# way, way*query_shot,r,d
        Q_bar_w=Q_bar_w.view(way,-1,d).contiguous() # way, way*query_shot*r, d
        dist_w=(Q_bar_w-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
       


    
        recon_dist=weight_d/(weight_d+weight_h+weight_w)*dist_d+weight_h/(weight_d+weight_h+weight_w)*dist_h+weight_w/(weight_d+weight_h+weight_w)*dist_w
        #recon_dist=dist_d
        #recon_dist=weight_d/(weight_d+weight_h)*dist_d+weight_h/(weight_d+weight_h)*dist_h
        #recon_dist=dist_w
        #recon_dist=weight_h/(weight_w+weight_h)*dist_h+weight_w/(weight_w+weight_h)*dist_w
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        Q_bar_d=Q_bar_d.view(way,-1,resolution,d)  # way, way*query_shot,resolution, d
        Q_bar_d=Q_bar_d.permute(1,0,2,3) #  way*query_shot, way, resolution, d
        Q_bar_h=Q_bar_h.view(way,-1,resolution,d)  # way, way*query_shot,resolution, d
        Q_bar_h=Q_bar_h.permute(1,0,2,3) #  way*query_shot, way, resolution, d
        Q_bar_w=Q_bar_w.view(way,-1,resolution,d)  # way, way*query_shot,resolution, d
        Q_bar_w=Q_bar_w.permute(1,0,2,3) #  way*query_shot, way, resolution, d
       
       
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist
   

    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist= self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index
    

    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction


    def forward(self,inp):

        neg_l2_dist, support= self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
            
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction, support
