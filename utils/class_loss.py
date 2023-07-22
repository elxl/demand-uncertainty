import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .distributions import zipoisson_prob_log


class MVELoss(nn.Module):
    def __init__(self,dist):
        super().__init__()
        self.dist = dist
        
    def forward(self, output_loc, output_scale=None, target=None, output_pi=None):
        
        loc = torch.flatten(output_loc)
        if output_scale is not None:
            scale = torch.flatten(output_scale)
        if output_pi is not None:
            pi = torch.flatten(output_pi)
        t = torch.flatten(target)

        if self.dist == 'laplace':
            d = torch.distributions.laplace.Laplace(loc, scale)
            loss = d.log_prob(t)

        elif self.dist == 'tnorm':
            d = torch.distributions.normal.Normal(loc, scale)
            prob0 = d.cdf(torch.Tensor([0]).to(target.device))
            loss = d.log_prob(t) - torch.log(1-prob0)

        elif self.dist == 'lognorm':
            d = torch.distributions.log_normal.LogNormal(loc, scale)
            loss = d.log_prob(t+0.000001)
            
        elif self.dist == 'poisson':
            d = torch.distributions.poisson.Poisson(loc)
            loss = d.log_prob(t)

        elif self.dist == 'zipoisson':
            # zipped = torch.cat([loc.unsqueeze(0),scale.unsqueeze(0),t.unsqueeze(0)],dim=0)
            # loss = torch.stack([
            #     zipoisson_prob_log(x_i[0],x_i[1],x_i[2]) for i, x_i in enumerate(torch.unbind(zipped, dim=1), 0)
            # ])
            d = torch.distributions.poisson.Poisson(loc)
            loss = torch.log(1-scale) + d.log_prob(t)

        elif (self.dist == 'norm') | (self.dist == 'norm_homo'):
            d = torch.distributions.normal.Normal(loc, scale)
            loss = d.log_prob(t)

        elif self.dist == 'nb':
            d = torch.distributions.NegativeBinomial(loc,scale)           
            loss = d.log_prob(t)

        elif self.dist == 'zinb':
            pass
        else:
            print("Dist error")
            return 0

       
        return -torch.sum(loss)