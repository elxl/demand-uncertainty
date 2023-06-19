import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#---------------GCN layer---------------#

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        Input size: (batch,nstations,in_features)
        adj size: (batch,nstations,nstations)
        output size: (batch,nstations,out_features)
        
        """
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), -1, -1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_features} -> {self.out_features})'
    
class GCNLayer(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.5):
        super(GCNLayer, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nout)

        self.dropout = dropout

    def forward(self, x, adj):
        """
        adj: Adjancency matrix tensor. Shape n_station,n_station
        """

        # Convert adjacency matrix to edge index and edge weights
        edge = adj.nonzero()
        edge_index = edge.t().contiguous()
        edge_weight = adj[edge[:,0],edge[:,1]]

        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.gc2(x, adj)
        # out = F.log_softmax(x, dim=1)
        return out