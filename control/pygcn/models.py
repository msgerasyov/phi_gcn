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
        G = nx.from_numpy_matrix(adj.data.cpu().numpy())
        g = dgl.from_networkx(G)
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return F.log_softmax(x, dim=1)
