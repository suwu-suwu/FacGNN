from fastNLP.modules.torch import VarGRU

import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import math
import dgl
import torch.nn.functional as F
# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.model import PointerGeneratorNetworks

class HSumGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed,vocab2):

        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim
        self.matrix_tree_layer = Matrix_Tree_Layer()

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)
        self.n_feature_proj1 = nn.Linear(hps.n_feature_size , hps.hidden_size, bias=False)
        #self.actor = Model(8, 8, 8, 8 ,3)
        self.model1 = PointerGeneratorNetworks(vob_size=50000,embed_dim=64,hidden_dim=128,
                                     pad_idx = vocab2.pad_idx,dropout=0.5,pointer_gen = True,
                                     use_coverage=True,_embed=self._embed)
        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,#输入的维度
                                out_dim=hps.hidden_size,#输出的维度
                                num_heads=hps.n_head,#多头注意力参数
                                attn_drop_out=hps.atten_dropout_prob,#梯度裁剪
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,#中间层参数
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,#特征嵌入的大小 
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        self.sent2sent = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=hps.hidden_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2S"
                                )

        self.sent2T = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=hps.hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2T"
                                )
        self.T2sent = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=hps.hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="T2S"
                                )
        # node classification
        self.n_feature = hps.hidden_size
        self.wh = nn.Linear(self.n_feature, 2)
        self.wh1 = nn.Linear(64, 256)
        #self.SAttention=LSTM_Attention(256,1,256)
    def forward(self, graph,f1,leixin):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :return: result: [sentnum, 2]
        """
       
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
         # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        a,b,(h,c),unpacked_len=self.set_snfeature(graph)
        
        sent_feature = self.n_feature_proj(a)    # [snode, n_feature_size]
        T_feature = self.n_feature_proj(b)    # [snode, n_feature_size]

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)
            #T_feature = self.sent2T(graph, T_feature, sent_state)

            #sent_state = self.T2sent(graph, T_feature, sent_state)
            #sent_state = self.sent2sent(graph, sent_state, sent_state)
        
        
        
        #print(sent_state.shape,"fscsdfcvsdvcsfs")
        #word_state = self.sent2word(graph, word_state, sent_state)
        #sent_state = self.word2sent(graph, word_state, sent_state)
        #sent_state = self.ffnn(sent_state).squeeze(0)
        sent_statee=sent_state
        sent_state1=self.wh1(sent_statee)
        split_lstm_feature = []
        start = 0
        for length in unpacked_len:
            end = start + length
            split_lstm_feature.append(sent_state1[start:end])
            start = end
        
        
        
        
        reconstructed_unpacked = rnn.pad_sequence(split_lstm_feature, batch_first=True)
        
        
        result = self.wh(sent_state)
        if leixin in ["train"]:
            inputs = {'encoder_input':f1[0],
                  'encoder_mask':f1[1],
                  'encoder_with_oov':f1[2],
                  'oovs_zero':f1[3],
                  'context_vec':f1[4],
                  'coverage':f1[5],
                  'decoder_input':f1[6],
                  'decoder_mask':f1[7],
                  'decoder_target':f1[8],
                  'reconstructed_unpacked':reconstructed_unpacked,
                  'h':(h,c)}
            print(reconstructed_unpacked.shape,"reconstructed_unpacked")
            print(h.shape,"h")
            result =self.model1(**inputs)
            
        if leixin in ["eval"]:
            #self.model1.eval()
            result = self.model1(encoder_input = f1[0],
                        encoder_mask= f1[1],
                        encoder_with_oov = f1[2],
                        oovs_zero = f1[3],
                        context_vec = f1[4],
                        coverage = f1[5],
                        mode = "eval",
                        beam_size = 10,
                        reconstructed_unpacked=reconstructed_unpacked,
                        h=(h,c)
                             )
        #loss,result =self.model1(**inputs)
        return result
    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        
        self.sent_pos_embed2 = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(300 + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        self.gru=VarGRU(input_size=self.embed_size, hidden_size=self.lstm_hidden_state, num_layers=self._hps.lstm_layers, input_dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        #self.ngram_enc = sentEncoder(self._hps, self._embed)
        self.ngram_enc = sentEncoder(self._hps, self._embed)
        self.cnnronghe = nn.Linear(self._hps.n_feature_size*2, self._hps.n_feature_size)
    def _sent_cnn_feature(self, graph, snode_id):   #句子做cnn  局部特征提取
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        #graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        
        #position= torch.arange(1, len(snode_id)+1).view(-1, 1).long().cuda() 
        position_embedding = self.sent_pos_embed(snode_pos)
        
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        '''cnn_feature1 = self.cnn_proj(input2cnn_feature + position_embedding)
        graph.nodes[snode_id].data["input_cnn_feature"] = cnn_feature1'''
        
        return cnn_feature
    
    def _entity_cnn_feature(self, graph, snode_id):   #句子做cnn  局部特征提取
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        
        #snode_pos= torch.arange(1, len(snode_id)+1).view(-1, 1).long().cuda() 
        
        position_embedding = self.sent_pos_embed2(snode_pos.view(-1) )
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        
        '''cnn_feature1 = self.cnn_proj(input2cnn_feature + position_embedding)
        graph.nodes[snode_id].data["input_cnn_feature"] = cnn_feature1'''
        
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):  #句子做rnn 全局特征提取
        
        
        pad_seq = rnn.pad_sequence(features, batch_first=True)#填充  默认batch_size在第一维度
        
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True) #需要保证，LSTM处理之后，这个batch还是保持原来的序列
        lstm_output, (h,c) = self.lstm(lstm_input)
        #print(lstm_output.shape,"unpacked")
        
        #lstm_output, _ = self.gru(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        
        
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        #print(unpacked.shape,"hjbiuhbi")
        
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))
        
        #lstm_feature=self.SAttention(lstm_feature)
        #lstm_feature = self.lstm_proj(lstm_feature)  # [n_nodes, n_feature_size]
        return lstm_feature,(h,c),unpacked_len
    
    def _entity_lstm_feature(self, features, glen):  #句子做rnn 全局特征提取
        
        pad_seq = rnn.pad_sequence(features, batch_first=True)#填充  默认batch_size在第一维度
        
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True,enforce_sorted=False) #需要保证，LSTM处理之后，这个batch还是保持原来的序列,enforce_sorted=False
        #lstm_output, _ = self.lstm(lstm_input)
        lstm_output, _ = self.gru(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        #print(torch.cat(lstm_embedding, dim=0).shape,"hjbiuhbi")
        
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0)) 
        #lstm_feature=self.SAttention(lstm_feature)
        #lstm_feature = self.lstm_proj(lstm_feature)  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   
        Tedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 2)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        Tfaet=graph.edges[Tedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        graph.edges[Tedge_id].data["tfidfembed"] = self._TFembed(Tfaet)
        return w_embed
    '''def settosetEdgfeature(self,graph):
        edgid = graph.filter_edges(lambda edges: edges.data["dtype"] == 1) 
        setnid=graph.find_edges(edgid)
        a=graph.nodes[setnid[0]].data["sent_embedding"]
        b=graph.nodes[setnid[1]].data["sent_embedding"]
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(a,b)
        #output = torch.norm(a-b, dim=1, p=2)
        
        graph.edges[edgid].data["embed"] = output.reshape(-1, 1)
        return '''
    def settosetEdgfeature(self,graph1):
        feature=[]
        glist = dgl.unbatch(graph1)
        
        for graph in glist:
            snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            
            sent_embedding=graph.nodes[snode_id].data["sent_embedding"]
            d, d_0 = self.matrix_tree_layer(sent_embedding.view(1, -1, 300),torch.tensor([1]).cuda())
            
            feature.extend(d_0[0])
            #graph.nodes[snode_id].data["root"] = torch.as_tensor(d_0[0]).cuda()
        val= torch.tensor(feature).cuda()

        return val   
    def set_snfeature(self, graph):
        # node feature
        
        
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        snode_id1 = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        cnn_feature1= self._entity_cnn_feature(graph, snode_id1)


        features, glen,features1, glen1 = get_snode_feat(graph, feat="sent_embedding")
        
        
        lstm_feature,(h,c),unpacked_len = self._sent_lstm_feature(features, glen)
        lstm_feature1 = self._entity_lstm_feature(features1, glen1)
        
        entity_root=self.settosetEdgfeature(graph)
        graph.nodes[snode_id].data["root"] =entity_root
        
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        topi_feature = torch.cat([cnn_feature1, lstm_feature1], dim=1)  # [n_nodes, n_feature_size * 2]


        return node_feature,topi_feature,(h,c),unpacked_len




class HSumDocGraph(HSumGraph):
    """
        without sent2sent and add residual connection
        add Document Nodes
    """

    def __init__(self, hps, embed):
        super().__init__(hps, embed)
        self.dn_feature_proj = nn.Linear(hps.hidden_size, hps.hidden_size, bias=False)
        self.wh = nn.Linear(self.n_feature * 2, 2)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, type=0
                word2doc, doc2word: tffrac=int, type=0
                sent2doc: type=2
        :return: result: [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # sent and doc node init
        graph.nodes[snode_id].data["init_feature"] = sent_feature
        doc_feature, snid2dnid = self.set_dnfeature(graph)
        doc_feature = self.dn_feature_proj(doc_feature)
        graph.nodes[dnode_id].data["init_feature"] = doc_feature

        # the start state
        word_state = word_feature
        sent_state = graph.nodes[supernode_id].data["init_feature"]
        sent_state = self.word2sent(graph, word_state, sent_state)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        # extract sentence nodes
        s_state_list = []
        for snid in snode_id:
            d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state = torch.cat([s_state, d_state], dim=-1)
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        result = self.wh(s_state)
        return result


    def set_dnfeature(self, graph):
        """ init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_feature_list = []
        snid2dnid = {}
        for dnode in dnode_id:
            snodes = [nid for nid in graph.predecessors(dnode) if graph.nodes[nid].data["dtype"]==1]
            doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
            assert not torch.any(torch.isnan(doc_feature)), "doc_feature_element"
            node_feature_list.append(doc_feature)
            for s in snodes:
                snid2dnid[int(s)] = dnode
        node_feature = torch.stack(node_feature_list)
        return node_feature, snid2dnid


def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    
    feature = []
    glen = []
    feature1 = []
    glen1 = []
    for g in glist:
        
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        snode_id1 = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
        feature1.append(g.nodes[snode_id1].data[feat])
        glen1.append(len(snode_id1))
        
            
    return feature, glen,feature1, glen1




class Matrix_Tree_Layer(nn.Module):
    def __init__(self):
        super(Matrix_Tree_Layer, self).__init__()
        #self.args = args
        self.str_dim_size = 300
        self.h_dim = 300
        
        # Projection for parent and child representation
        self.tp_linear = nn.Linear(self.h_dim, self.str_dim_size, bias=True)
        self.tc_linear = nn.Linear(self.h_dim, self.str_dim_size, bias=True)
        self.bilinear = BilinearMatrixAttention(self.str_dim_size, self.str_dim_size, use_input_biases=False, label_dim=1)
        
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        
    def forward(self, sent_vecs, enc_sent_padding_mask):
        
        batch_size = sent_vecs.shape[0]
        sent_num = sent_vecs.shape[1]
        
        tp = torch.relu(self.tp_linear(sent_vecs))#父表示
        tc = torch.relu(self.tc_linear(sent_vecs))#子表示
        
        
        #print(sent_vecs.shape,sent_vecs)
        
        # Using the bilinear attention to compute f_jk: fjk = u^T_k W_a u_j
        scores = self.bilinear(tp, tc).view(batch_size, sent_num, sent_num)
        root = self.fi_linear(tp).view(batch_size, sent_num)  # 句子重要程度概率，将父表示乘上可学习矩阵  W*tp

        # masking out diagonal elements, see Eqt 2.1
        mask = scores.new_ones((scores.size(1), scores.size(1))) - scores.new_tensor(torch.eye(scores.size(1), scores.size(1))).cuda()

        mask = mask.unsqueeze(0).expand(scores.size(0), mask.size(0), mask.size(1))
        
        '''if self.args.not_sparse:
            A_ij = torch.exp(torch.tanh(scores))
            A_ij = (A_ij.transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1)).transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1) + 1e-6
            A_ij = A_ij * mask
            f_i = (torch.exp(torch.tanh(root)) * enc_sent_padding_mask) + 1e-6'''
        
        A_ij = torch.relu(scores)
        
        
        
        enc_sent_padding_mask=enc_sent_padding_mask.cuda()
        A_ij = (A_ij.transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1)).transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1) + 1e-6
        A_ij = A_ij * mask  #得到邻接矩阵
        f_i = (torch.relu(root) * enc_sent_padding_mask) + 1e-6

        tmp = torch.sum(A_ij, dim=1)
        res = A_ij.new_zeros((batch_size, sent_num, sent_num)).cuda() #.to(self.device)
        #tmp = torch.stack([torch.diag(t) for t in tmp])
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)

        L_ij = -A_ij + res   #A_ij has 0s as diagonals
        
        L_ij_bar = L_ij.clone()
        
        L_ij_bar[:,0,:] = f_i
        
        LLinv = None
        LLinv = torch.inverse(L_ij_bar)
        
        d0 = f_i * LLinv[:,:,0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1,2) * LLinv_diag ).transpose(1,2)
        tmp2 = A_ij * LLinv.transpose(1,2)

        temp11 = A_ij.new_zeros((batch_size, sent_num, 1))
        temp21 = A_ij.new_zeros((batch_size, 1, sent_num))

        temp12 = A_ij.new_ones((batch_size, sent_num, sent_num-1))
        temp22 = A_ij.new_ones((batch_size, sent_num-1, sent_num))

        mask1 = torch.cat([temp11,temp12],2).cuda() #.to(self.device)
        mask2 = torch.cat([temp21,temp22],1).cuda() #.to(self.device)

        # Eqt: P(zjk = 1) = (1 − δ(j, k))AjkL¯−1kk − (1 − δ(j, 1))AjkL¯−1
        dx = mask1 * tmp1 - mask2 * tmp2
        
        return dx, d0 

    
class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.
    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
        activation.
    use_input_biases : ``bool``, optional (default = False)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    label_dim : ``int``, optional (default = 1)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
        for matrices, rather than vectors.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False,
                 label_dim: int = 1) -> None:
        super(BilinearMatrixAttention, self).__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self._bias = Parameter(torch.Tensor(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        #围绕着 方向依赖的边概率在计算  具体计算公式： tp^T*W*tc(i句子的父表示与j句子的子表示)
        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1)
