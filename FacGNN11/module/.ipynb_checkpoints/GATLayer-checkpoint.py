

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################### SubLayer #########################################
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output


######################################### HierLayer #########################################

class SGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim , 1, bias=False)
        self.feat_fc = nn.Linear(1, out_dim, bias=False)
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["embed"]) 
        z2 = torch.cat([edges.src['z'], edges.dst['z'],dfeat], dim=1)  # [edge_num, 2 * out_dim]
        
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h,o):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 1)
        z = self.fc(h)
        z1 = self.fc1(o)
        g.nodes[wnode_id].data['z'] = z1
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc1 = nn.Linear(64, out_dim, bias=False)
        
        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear( out_dim, 1, bias=False)
        self.weight_gate = nn.Linear(2 * out_dim, out_dim)
        
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])                  # [edge_num, out_dim]

        root=edges.dst['root'].view(-1,1)
        
        z2=root*edges.src['z']+ (1-root)*edges.dst['z']
        
        gate = self.weight_gate(torch.cat((z2, edges.dst['z']), -1))
        
        gate = torch.sigmoid(gate)
        
        z22 = gate * torch.tanh(z2) + (1 - gate) * edges.dst['z']
        z222=z22+edges.src['z']+ dfeat
        
       
        
        z3=F.leaky_relu(z222)
        wa = self.attn_fc(z3)
        return {'e': wa}

    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h,o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        # print("id in WSGATLayer")
        # print(wnode_id, snode_id, wsedge_id)
        z = self.fc(h)
        z1 = self.fc1(o)
        g.nodes[snode_id].data['z'] = z1
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]



class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc1 = nn.Linear(300, out_dim, bias=False)
        
        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear( out_dim, 1, bias=False)
        self.weight_gate = nn.Linear(2 * out_dim, out_dim)
        
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        
        #a,b= edges.dst['z'].size()
        #senet=SELayer(channel=a,reduction=12)
        #dst=senet(edges.dst['z'])
        '''root=edges.src['root'].view(-1,1)
        z2=(1-root)*edges.src['z']+ root*edges.dst['z']#+ dfeat
        gate = self.weight_gate(torch.cat((z2, edges.src['z']), -1))
        gate = torch.sigmoid(gate)
        z22 = gate * torch.tanh(z2) + (1 - gate) * edges.src['z']#+ dfeat'''
        
        z2=edges.src['z']+ edges.dst['z']+ dfeat
        #z2=edges.src['z']+ dst+ dfeat
        z3=F.leaky_relu(z2)
        wa = self.attn_fc(z3)
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h,o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        z1 = self.fc1(o)
        g.nodes[wnode_id].data['z'] = z1
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]
    
class TSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc1 = nn.Linear(64, out_dim, bias=False)
        
        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear( out_dim, 1, bias=False)
        self.weight_gate = nn.Linear(2 * out_dim, out_dim)
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])                  # [edge_num, out_dim]

        '''a,b= edges.src['z'].size()
        senet=SELayer(channel=a,reduction=12)
        src=senet(edges.src['z'])'''
        
        root=edges.dst['root'].view(-1,1)
        z2=root*edges.src['z']+ (1-root)*edges.dst['z']#+ dfeat
        gate = self.weight_gate(torch.cat((z2, edges.dst['z']), -1))
        gate = torch.sigmoid(gate)
        z22 = gate * torch.tanh(z2) + (1 - gate) * edges.dst['z']
        z222=z22+edges.src['z']+ dfeat
        
        
        
        z3=F.leaky_relu(z222)
        wa = self.attn_fc(z3)
        return {'e': wa}

    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h,o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
      
        
        z = self.fc(h)
        z1 = self.fc1(o)
        g.nodes[snode_id].data['z'] = z1
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]
    
class STGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc1 = nn.Linear(64, out_dim, bias=False)
        
        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(  out_dim, 1, bias=False)
        self.weight_gate = nn.Linear(2 * out_dim, out_dim)
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        #z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        #print(edges.src['root'])
        root=edges.dst['root'].view(-1,1)
        z2=root*edges.src['z']+ (1-root)*edges.dst['z']#+ dfeat
        gate = self.weight_gate(torch.cat((z2, edges.dst['z']), -1))
        gate = torch.sigmoid(gate)
        z22 = gate * torch.tanh(z2) + (1 - gate) * edges.dst['z']
        z222=z22+edges.src['z']+ dfeat
        
        z3=F.leaky_relu(z222)
        wa = self.attn_fc(z3)
        '''root=edges.src['root'].view(-1,1)
        z2=(1-root)*edges.src['z']+ root*edges.dst['z']#+ dfeat
        gate = self.weight_gate(torch.cat((z2, edges.src['z']), -1))
        gate = torch.sigmoid(gate)
        z22 = gate * torch.tanh(z2) + (1 - gate) * edges.src['z']'''
        

        
        #wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h,o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        z1 = self.fc1(o)
        g.nodes[wnode_id].data['z'] = z1
        g.nodes[snode_id].data['z'] = z
        
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]

    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=12):
        super(SELayer, self).__init__()


        '''self.fc = nn.Sequential(
            # channel // reduction:减少计算量
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 变成原来的通道数
            nn.Linear(channel // reduction, channel, bias=False),
            # 将结果值映射到[0,1]的区间
            nn.Sigmoid())'''
        self.fc1=nn.Linear(channel, channel // reduction, bias=False).cuda()
        self.relu=nn.ReLU(inplace=True).cuda()
        self.fc2=nn.Linear(channel // reduction, channel, bias=False).cuda()
        self.sg=nn.Sigmoid().cuda()
    def forward(self, x):
        
        #y = self.avg_pool(x)
        y=torch.mean(x,dim=-1).view(1,-1).cuda()
        #print(y,"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        #print(y.shape)
        # [B,C]=>self.fc(y)=>[B,C/2]=>
        y = self.relu(self.fc1(y))
        y = self.sg(self.fc2(y))
        
        # 将计算所得权重与原先张量相乘
        
        return   torch.mul(x,torch.unsqueeze(y,dim=2)).squeeze(0)
