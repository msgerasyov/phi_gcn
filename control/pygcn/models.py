import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import dgl
from dgl.nn import GATConv, SAGEConv, GraphConv
import networkx as nx


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GCN, self).__init__()

        self.gc1= GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        model_device = next(self.parameters()).device
        G = nx.from_numpy_matrix(adj.data.cpu().numpy())
        g = dgl.from_networkx(G).to(model_device)
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, num_heads=8)
        self.conv2 = GATConv(8 * nhid, nclass, num_heads=1)

    def forward(self, x, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        model_device = next(self.parameters()).device
        G = nx.from_numpy_matrix(adj.data.cpu().numpy())
        g = dgl.from_networkx(G).to(model_device)
        h = self.conv1(g, x)
        h = h.view(-1, h.shape[1] * h.shape[2])
        h = F.elu(h)
        h = self.conv2(g, h)
        h = h.view(-1, h.shape[1] * h.shape[2])
        return F.log_softmax(h, dim=1)


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(SAGE, self).__init__()

        self.gc1= SAGEConv(nfeat, nhid, 'mean', bias=True)
        self.gc2 = SAGEConv(nhid, nclass, 'mean', bias=True)

    def forward(self, x, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        model_device = next(self.parameters()).device
        G = nx.from_numpy_matrix(adj.data.cpu().numpy())
        g = dgl.from_networkx(G).to(model_device)
        x = F.relu(self.gc1(g, x))
        x = self.gc2(g, x)
        return F.log_softmax(x, dim=1)
