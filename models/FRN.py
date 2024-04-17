import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import Conv_4,ResNet
from torchvision.utils import save_image
import torchvision.utils as vutils
import os
from torch import Tensor

def pdist(x,y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist
class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)
            self.dim = num_channel*5*5
        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25 

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(4),requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True) 
            self.is_pretraining=True
        else:
            self.is_pretraining=False
    

    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        
        return feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous()# N,HW,C
    
    def get_feature_vector(self,inp):
        
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        if self.resnet:
            feature_map = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
            feature_vector = feature_map.view(batch_size,self.dim)
        else:
            feature_vector = feature_map.view(batch_size,self.dim)
        
        return feature_vector
    def get_proto_sim(self,inp,way,shot,query_shot):
        
        feature_vector = self.get_feature_vector(inp) 

        support = feature_vector[:way*shot].view(way,shot,self.dim)
        centroid = torch.mean(support,1) # way,dim
        query = feature_vector[way*shot:] # way*query_shot,dim
        neg_l2_dist = pdist(query,centroid).neg().view(way*query_shot,way) #way*query_shot,way
        centroid_norm=centroid.norm(dim=1).unsqueeze(1)
        centroid=centroid/(centroid_norm+1e-8)
        sim=torch.mm(centroid,centroid.T)
        label_matrix=torch.eye(sim.shape[0],sim.shape[1]).cuda()
        mask=label_matrix
        
        support_sim=sim.masked_select((torch.ones_like(mask)-mask).bool()).mean()
        return neg_l2_dist



    def get_recon_dist(self,query,support,alpha,beta,Woodbury=False):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not
      
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

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        
        return dist
    
    def get_recon_dist_oppose(self,query,support,alpha,beta,Woodbury=False,test_mode=False):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not
        
        support0=support
        
        if test_mode:
            way=5
            query_shot=15
        else:
            way=self.way
            query_shot=self.shots[1]
        resolution=self.resolution
        d=self.d
        query=query.view(way*query_shot,resolution,d)
        support0=support0.view(-1,d)
        #convert
        support=query
        query=support0        

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
        
        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        

        return dist
    
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False,test_mode=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        alpha1 = self.r[2]
        beta1 = self.r[3]
        feature_map = self.get_feature_map(inp)#inp:75*3*84*84 输出：75*25*64
        
        support = feature_map[:way*shot].view(way, shot*resolution , d) #5*125*64
        query = feature_map[way*shot:].view(way*query_shot*resolution, d) #1250*64
       
        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        recon_dist_oppose=self.get_recon_dist_oppose(query=query,support=support,alpha=alpha1,beta=beta1,test_mode=test_mode) # way*shot*resolution, way*query_shot
        neg_l2_dist_forward = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        neg_l2_dist_oppose=recon_dist_oppose.neg().view(way,shot*resolution,way*query_shot).mean(1).permute(1,0) 
     
        if return_support:
            return neg_l2_dist_forward,neg_l2_dist_oppose, support
        else:
            return neg_l2_dist_forward,neg_l2_dist_oppose
   
    def get_neg_l2_dist_pretrain(self,inp,way,shot,query_shot,return_support=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        alpha1 = self.r[2]
        beta1 = self.r[3]
        feature_map = self.get_feature_map(inp)#inp:75*3*84*84 输出：75*25*64
        
        support = feature_map[:way*shot].view(way, shot*resolution , d) #5*125*64
        query = feature_map[way*shot:].view(way*query_shot*resolution, d) #1250*64
        
        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        neg_l2_dist_forward = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way

     
        if return_support:
            return neg_l2_dist_forward, support
        else:
            return neg_l2_dist_forward

    def meta_test(self,inp,way,shot,query_shot,test_mode=False):

        if self.is_pretraining:
            neg_l2_dist_forward = self.get_neg_l2_dist_pretrain(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)
            _,max_index = torch.max(neg_l2_dist_forward,1)
        else:
        
            neg_l2_dist_forward,neg_l2_dist_oppose = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot,test_mode=test_mode)
            _,max_index = torch.max(neg_l2_dist_forward+0*neg_l2_dist_oppose,1)
        

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

        neg_l2_dist_forward,neg_l2_dist_oppose, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
    
        

        logits_forward = neg_l2_dist_forward*self.scale
        log_prediction_forward = F.log_softmax(logits_forward,dim=1)

        logits_oppose = neg_l2_dist_oppose*self.scale
        log_prediction_oppose = F.log_softmax(logits_oppose,dim=1)
      
        return log_prediction_forward, log_prediction_oppose,support
    
if __name__ == '__main__':
    train_way=5
    train_shot=5
    train_query_shot=1
    model = FRN(way=train_way,
            shots=[train_shot, train_query_shot],
            )
    data = torch.randn(30, 3, 84, 84)
    log_prediction, support = model(data)
    print(log_prediction.shape)
    print(support.shape)
    '''
    fig_dir = './imgs' 
    os.makedirs(fig_dir, exist_ok=True)
    cur_iter = 0
    num_row=15
    data0=torch.randn(1, 3, 84, 84)
    vutils.save_image(data0.data.cpu(),'{}/image_one_15_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)
    '''