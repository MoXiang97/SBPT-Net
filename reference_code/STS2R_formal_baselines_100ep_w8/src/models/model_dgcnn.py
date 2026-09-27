import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    k = min(int(k), int(x.size(2)))
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, knn_source=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        graph_input = x if knn_source is None else knn_source
        idx = knn(graph_input, k=k)   # (batch_size, num_points, k)
    neighbor_count = idx.size(-1)
    
    device = torch.device('cuda') if x.is_cuda else torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, neighbor_count, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, neighbor_count, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature # (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    def __init__(self, args=None, num_classes=2, k=20, emb_dims=1024, dropout=0.5, input_channels=6):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        
        # Input channels calculation for EdgeConv1
        # get_graph_feature returns 2 * input_channels features
        edge_in_channels = input_channels * 2
        
        # EdgeConv 1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(edge_in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 2
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 3
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # EdgeConv 4
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Global Feature Aggregation
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # Segmentation Head
        self.bn6 = nn.BatchNorm1d(256)
        self.conv6 = nn.Sequential(nn.Conv1d(emb_dims + 512, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn7 = nn.BatchNorm1d(128)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv8 = nn.Conv1d(128, num_classes, kernel_size=1, bias=False)
        self.dp1 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x: (batch_size, 6, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # The initial graph is spatial (XYZ only), while its edge features carry
        # the shared XYZRGB input used by every baseline. Later graphs are dynamic
        # and are rebuilt from learned point features.
        
        # Layer 1
        x = get_graph_feature(x, k=self.k, knn_source=x[:, :3, :]) # (B, 12, N, k)
        x = self.conv1(x) # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)
        
        # Layer 2
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        # Layer 3
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        # Layer 4
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        # Concatenate local features
        x = torch.cat((x1, x2, x3, x4), dim=1) # (B, 64+64+128+256=512, N)
        
        # Global Feature
        x = self.conv5(x) # (B, 1024, N)
        x_max = x.max(dim=2, keepdim=True)[0] # (B, 1024, 1)
        x_max = x_max.repeat(1, 1, num_points) # (B, 1024, N)
        
        # Concatenate global and local features (Hypercolumn)
        x = torch.cat((x_max, x1, x2, x3, x4), dim=1) # (B, 1024+512, N)
        
        # Segmentation Head
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        # Return raw logits [B, num_classes, N] for CrossEntropyLoss
        return x, None
