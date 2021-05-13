import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import dgl
from dgl.nn import GATConv, SAGEConv, GraphConv
import networkx as nx


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GCN, self).__init__()

        self.gc1= GraphConv(nfeat, nhid, norm='both')
        self.gc2 = GraphConv(nhid, nclass, norm='both')
        self.dropout = dropout

    def forward(self, x, g):
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, num_heads=4)
        self.conv2 = GATConv(4 * nhid, nclass, num_heads=1)

    def forward(self, x, g):
        h = self.conv1(g, x)
        h = h.view(-1, h.shape[1] * h.shape[2])
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.view(-1, h.shape[1] * h.shape[2])
        return F.log_softmax(h, dim=1)


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(SAGE, self).__init__()

        self.gc1= SAGEConv(nfeat, nhid, aggregator_type='mean', bias=True)
        self.gc2 = SAGEConv(nhid, nclass, aggregator_type='lstm', bias=True)

    def forward(self, x, g):
        x = F.elu(self.gc1(g, x))
        x = self.gc2(g, x)
        return F.log_softmax(x, dim=1)
