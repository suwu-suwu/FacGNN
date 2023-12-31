import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import code

class Beam(object):
    def __init__(self,tokens,log_probs,status,context_vec,coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.status = status
        self.context_vec = context_vec
        self.coverage = coverage

    def update(self,token,log_prob,status,context_vec,coverage):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            status = status,
            context_vec = context_vec,
            coverage = coverage
        )


    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


def sort_beams(beams):
    return sorted(beams, key=lambda beam:beam.avg_log_prob, reverse=True)


class Encoder(nn.Module):
    def __init__(self,vob_size,embed_dim,hidden_dim,layer_num = 2,pad_idx = 0,dropout = 0.5,_embed= None):
        super(Encoder,self).__init__()
        self._embed=_embed

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(num_embeddings=vob_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=self.pad_idx)

        self.lstm = nn.LSTM(input_size = 300,
                            hidden_size = hidden_dim,
                            num_layers = layer_num,
                            dropout = dropout,
                            batch_first = True,
                            bidirectional = True)

        self.dropout = nn.Dropout(p=dropout)


    # x.shape (batch,seq_len)  词的索引
    # mask.shape (batch,seq_len)   每个样本的真实长度
    def forward(self,x,mask) :
        x=x.to("cuda:0")

        mask=mask
        embedded=self._embed(x)
        #embedded = self.embedding(x)
        
        
        embedded = self.dropout(embedded)

        seq_lens = mask.sum(dim = -1)
        
        packed = pack_padded_sequence(input=embedded,lengths=seq_lens,batch_first=True,enforce_sorted=False)
        output_packed,(h,c) = self.lstm(packed)
       

        # pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):

        output,_ = pad_packed_sequence(sequence = output_packed,
                                       batch_first = True,
                                       padding_value=self.pad_idx,
                                       total_length = seq_lens.max())

        return output,(h,c)



class Reduce(nn.Module):
    def __init__(self,hidden_dim,dropout = 0.5):
        super(Reduce,self).__init__()

        self.hidden_dim = hidden_dim

        self.reduce_h = nn.Linear(hidden_dim*4,hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim*4,hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,h,c):
        
        assert 4 == h.shape[0]
        assert 4 == c.shape[0]

        assert self.hidden_dim == h.shape[2]
        assert self.hidden_dim == c.shape[2]

        h = h.reshape(-1,self.hidden_dim*4)
        c = c.reshape(-1,self.hidden_dim*4)

        h_output = self.dropout(self.reduce_h(h))
        c_output = self.dropout(self.reduce_c(c))

        h_output = F.relu(h_output)
        c_output = F.relu(c_output)

        # h_output.shape = c_output.shape =
        # (batch,hidden)
        return h_output.unsqueeze(0),c_output.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self,hidden_dim,use_coverage = False):
        super(Attention, self).__init__()

        self.use_coverage = use_coverage

        self.w_h = nn.Linear(hidden_dim*2,hidden_dim*2,bias=False)     #
        self.w_s = nn.Linear(hidden_dim*2,hidden_dim*2,bias=False)

        if self.use_coverage:
            self.w_c = nn.Linear(1,hidden_dim*2)

        self.v = nn.Linear(hidden_dim*2,1,bias=False)

    # h ：encoder hidden states h_i ,On each step t. (batch,seq_len,hidden*2)
    # mask, 0-1 encoder_mask (batch,seq_len)
    # s : decoder state s_t,one step (batch,hidden*2)
    # coverage : sum of attention score (batch,seq_len)
    def forward(self,h,mask,s,coverage):

        encoder_feature = self.w_h(h)   # (batch,seq_len,hidden*2)
        decoder_feature = self.w_s(s).unsqueeze(1)  # (batch,1,hidden*2)
        
        # broadcast 广播运算
        attention_feature = encoder_feature + decoder_feature  # (batch,seq_len,hidden*2)
        
        #padding_steps = 400 - attention_feature.shape[1]
        #attention_feature = torch.cat([attention_feature, torch.zeros(32, padding_steps, 256).to("cuda:0")], dim=1)

        
    
        if self.use_coverage:
            coverage_feature = self.w_c(coverage.unsqueeze(2).to("cuda:0"))  # (batch,seq_len,hidden*2)
            
            attention_feature += coverage_feature


        e_t = self.v(torch.tanh(attention_feature)).squeeze(dim = 2)  # (batch,seq_len)

        mask_bool = (mask == 0).to("cuda:0")   # mask pad position eq True

        e_t.masked_fill_(mask = mask_bool,value= -float('inf'))

        a_t = torch.softmax(e_t,dim=-1)  # (batch,seq_len)

        if self.use_coverage:
            next_coverage = coverage.to("cuda:0")  + a_t.to("cuda:0") 

        return a_t,next_coverage


class GeneraProb(nn.Module):
    def __init__(self,hidden_dim,embed_dim):
        super(GeneraProb,self).__init__()

        self.w_h = nn.Linear(hidden_dim*2,1)
        self.w_s = nn.Linear(hidden_dim*2,1)
        self.w_x = nn.Linear(embed_dim,1)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self,h,s,x):
        h_feature = self.w_h(h)     # (batch,1)
        s_feature = self.w_s(s)     # (batch,1)
        x_feature = self.w_x(x)     # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p


class Decoder(nn.Module):
    def __init__(self,vob_size,embed_dim,hidden_dim,layer_num = 1,dropout = 0.5,pad_idx = 0,pointer_gen = True,
                 use_coverage= False,_embed= None):
        super(Decoder, self).__init__()

        self._embed=_embed
        self.pointer_gen = pointer_gen
        self.use_coverage = use_coverage

        self.embedding = nn.Embedding(num_embeddings=vob_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=pad_idx).to("cuda:0")
        

        self.get_lstm_input = nn.Linear(in_features=hidden_dim * 2 + 256,out_features=embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_num,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=False)

        self.attention = Attention(hidden_dim = hidden_dim,use_coverage=use_coverage)

        if pointer_gen:
            self.genera_prob = GeneraProb(hidden_dim = hidden_dim,
                                          embed_dim = embed_dim)

        self.dropout = nn.Dropout(p=dropout)


        # self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3,out_features=embed_dim),
        #                          self.dropout(),
        #                          nn.ReLU(),
        #                          nn.Linear(in_features=hidden_dim,out_features=vob_size),
        #                          self.dropout)
        self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3, out_features=hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=vob_size))
        
        self.dnco =nn.Linear(300 , 256, bias=False)

    # decoder_input_one_step (batch,1)
    # decoder_status = (h_t,c_t)  h_t (1,batch,hidden)
    # encoder_output (batch,seq_len,hidden*2)
    # encoder_mask (batch,seq_len)
    # context_vec (bach,hidden*2)  encoder weight sum about attention score
    # oovs_zero (batch,max_oov_size)  all-zero tensor
    # encoder_with_oov (batch,seq_len)  Index of words in encoder_with_oov can be greater than vob_size
    # coverage : Sum of attention at each step
    # step...
    def forward(self,decoder_input_one_step,decoder_status,encoder_output,
                encoder_mask,context_vec,oovs_zero,encoder_with_oov,coverage,step):
        
        #embed=self._embed(decoder_input_one_step.to("cuda:0"))
        #embed = self.embedding(decoder_input_one_step.to("cuda:0")) # (batch,embed_dim)
        embed=self._embed(decoder_input_one_step.to("cuda:0"))
        #print(embed.shape,context_vec.shape,"context_veccontext_veccontext_vec")
        embed=self.dnco(embed)
        #print(embed.shape,context_vec.shape,"context_vec")
        #print(torch.cat([context_vec.to("cuda:0"), embed], dim=-1).shape)
        x = self.get_lstm_input(torch.cat([context_vec.to("cuda:0"), embed], dim=-1)).unsqueeze(dim=1)  # (batch,1,hidden*2+embed_dim)
        #print(x.shape,"xxx")
        decoder_output,next_decoder_status = self.lstm(x,decoder_status)
        
        
        
        

        h_t,c_t = next_decoder_status

        batch_size = c_t.shape[1]

        h_t_reshape = h_t.reshape(batch_size,-1)
        c_t_reshape = c_t.reshape(batch_size,-1)

        status = torch.cat([h_t_reshape,c_t_reshape],dim = -1)   # (batch,hidden_dim*2)

        # attention_score (batch,seq_len)  Weight of each word vector
        # next_coverage (batch,seq_len)  sum of attention_score
        
        attention_score,next_coverage = self.attention(h = encoder_output,
                                                       mask = encoder_mask,
                                                       s = status,
                                                       coverage = coverage)

        # (batch,hidden_dim*2)  encoder_output weight sum about attention_score

        current_context_vec = torch.bmm(attention_score.unsqueeze(1),encoder_output).squeeze()
        if current_context_vec.shape[0]==256 :
            current_context_vec=current_context_vec.reshape(1,-1)
        # 尝试一下高级写法，结果和上面一行一致
        #current_context_vec = torch.einsum("ab,abc->ac",attention_score,encoder_output)

        # (batch,1)
        #if encoder_output.shape[0]==1 :
        #    print(current_context_vec.shape,status.shape,x.squeeze().shape,decoder_output.squeeze(dim = 1).shape,"sasqsqwsq")
        
        genera_p = None
        if self.pointer_gen:
            genera_p = self.genera_prob(h = current_context_vec,
                                        s = status,
                                        x = x.squeeze())

        # (batch,hidden_dim*3)
        #print(decoder_output.squeeze(dim = 1).shape,current_context_vec.shape,"current_context_veccurrent_context_veccurrent_context_vec")
        out_feature = torch.cat([decoder_output.squeeze(dim = 1),current_context_vec],dim = -1)

        # (batch,vob_size)
        output = self.out(out_feature)
        vocab_dist = torch.softmax(output,dim = -1)

        if self.pointer_gen:
            vocab_dist_p = vocab_dist * genera_p
            context_dist_p = attention_score * (1 - genera_p)
            if oovs_zero is not None:
                vocab_dist_p = torch.cat([vocab_dist_p.to("cuda:0"),oovs_zero.to("cuda:0")],dim = -1)
            final_dist = vocab_dist_p.scatter_add(dim = -1,index=encoder_with_oov.to("cuda:0"),src = context_dist_p.to("cuda:0"))
        else:
            final_dist = vocab_dist

        # code.interact(local = locals())
        return final_dist,next_decoder_status,current_context_vec,attention_score,genera_p,next_coverage




class PointerGeneratorNetworks(nn.Module):
    def __init__(self,vob_size = 50000,embed_dim = 128,hidden_dim = 128,pad_idx = 0,dropout = 0.5,pointer_gen = True,
                 use_coverage = True,eps = 1e-12,coverage_loss_weight = 1.0,unk_token_idx = 1,start_token_idx = 2,
                 stop_token_idx = 3,max_decoder_len = 170,min_decoder_len = 35,_embed= None):
        super(PointerGeneratorNetworks,self).__init__()
        self.all_mode = ["train","eval","decode"]
        self._embed=_embed
        self.vob_size = vob_size
        self.use_coverage = use_coverage
        self.eps = eps
        self.coverage_loss_weight = coverage_loss_weight
        self.max_decoder_len = max_decoder_len
        self.min_decoder_len = min_decoder_len
        self.start_token_idx = start_token_idx
        self.stop_token_idx = stop_token_idx
        self.unk_token_idx = unk_token_idx

        self.encoder = Encoder(vob_size = vob_size,embed_dim = embed_dim,hidden_dim = hidden_dim,pad_idx = pad_idx,
                               dropout = 0.5,_embed=self._embed)

        self.reduce = Reduce(hidden_dim = hidden_dim,dropout = dropout)


        self.decoder = Decoder(vob_size = vob_size,embed_dim = embed_dim,hidden_dim = hidden_dim,dropout = dropout,
                               pad_idx=pad_idx,pointer_gen = pointer_gen,use_coverage =use_coverage,_embed=self._embed)
        
        self.hidd1= nn.Linear(256 , 128, bias=False)
        self.hidd2= nn.Linear(256 , 128, bias=False)
        self.enco =nn.Linear(512 , 256, bias=False)

    # encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage, beam_size


    def forward(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,reconstructed_unpacked,h,decoder_input = None,decoder_mask = None,decoder_target = None,mode = "train",start_tensor = None,beam_size = 4):

        assert mode in self.all_mode
        if mode in ["train"]:
            return self._forward(encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,
                decoder_input,decoder_mask,decoder_target,reconstructed_unpacked,h)
        elif mode in ["decode"]:
            return self._decoder(encoder_input = encoder_input,encoder_mask = encoder_mask,
                                 encoder_with_oov = encoder_with_oov,oovs_zero = oovs_zero,context_vec = context_vec,
                                 coverage = coverage,beam_size = beam_size)
        elif mode in ["eval"]:
            return self._eval(encoder_input = encoder_input,encoder_mask = encoder_mask,
                                 encoder_with_oov = encoder_with_oov,oovs_zero = oovs_zero,context_vec = context_vec,
                                 coverage = coverage,beam_size = beam_size,reconstructed_unpacked=reconstructed_unpacked,h=h)

    # encoder_input=[16,400] 
    # encoder_mask=[16,400] 
    # encoder_with_oov=[16,400] 
    # oovs_zero=[16,13]
    # context_vec=[16,512]
    # coverage=[16,400] 全0
    # decoder_input=[16,100]
    # decoder_mask=[16,100]
    # decoder_target=[16,100]
    def _forward(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,decoder_input,decoder_mask,decoder_target,reconstructed_unpacked,h):
        beam_size = 10
        step = 0
        result = []
        # encoder_outputs=[32, 300, 256]    h [32, 29, 256]
        # encoder_hidden = ([4, 32, 128],[4, 32, 128])   [4, 32, 128]
        
        encoder_outputs1, encoder_hidden1 = self.encoder(encoder_input,encoder_mask)
        #print(encoder_outputs1.shape,encoder_hidden1[0].shape,"encoder_hidden1encoder_hidden1")
        mmm=reconstructed_unpacked.shape[1]
        encoder_outputs2 = F.pad(reconstructed_unpacked, (0, 0, 0, 470 - mmm, 0, 0))
        encoder_hidden2 =h
        #padding_steps = 300 - reconstructed_unpacked.shape[1]
        
        #print(encoder_outputs2.shape,encoder_hidden2[0].shape,"encoder_hidden2encoder_hidden2")
        encoder_outputs=self.enco(torch.cat([encoder_outputs1, encoder_outputs2], dim=2 ))
        
        
        encoder_hidden=(self.hidd1(torch.cat([encoder_hidden1[0], encoder_hidden2[0]], dim=2 )),self.hidd2(torch.cat([encoder_hidden1[1], encoder_hidden2[1]], dim=2)))
        #encoder_outputs = torch.cat([reconstructed_unpacked, torch.zeros(reconstructed_unpacked.shape[0], padding_steps, 256).to("cuda:0")], dim=1)
        
        #encoder_outputs =reconstructed_unpacked
        
        #print(encoder_hidden[0].shape,encoder_hidden[1].shape,"encoder_hidden")
        decoder_status = self.reduce(*encoder_hidden) # ([1,16,256],[1,16,256])

        decoder_lens = decoder_mask.sum(dim=-1) # 每个摘要的长度
        
        batch_max_decoder_len = decoder_lens.max() # 获取一个batch中摘要的最长长度
        
         
        assert batch_max_decoder_len <= self.max_decoder_len
        # code.interact(local = locals())
        #beams = [Beam(tokens = [self.start_token_idx],log_probs = [1.0],status=decoder_status,context_vec = context_vec,coverage = coverage)]
        
        all_step_loss = []
        for step in range(batch_max_decoder_len):
            decoder_input_one_step = decoder_input[:, step ]
            
            
            

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_input_one_step=decoder_input_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)

            target = decoder_target[:, step].unsqueeze(1) # 获取到目标位置的索引
            
            probs = torch.gather(final_dist.to("cuda:0"),dim=-1,index=target.to("cuda:0")).squeeze() # 获得这个词分布下，目标词的概率
            step_loss = -torch.log(probs + self.eps) # 计算概率

            if self.use_coverage:
                coverage_loss = self.coverage_loss_weight * torch.sum(torch.min(attention_score.to("cuda:0"), coverage.to("cuda:0")), dim=-1)
                step_loss += coverage_loss
                coverage = next_coverage


            all_step_loss.append(step_loss)


        token_loss = torch.stack(all_step_loss, dim=1)

        decoder_mask_cut = decoder_mask[:,:batch_max_decoder_len].float()
        assert decoder_mask_cut.shape == token_loss.shape


        token_loss_with_mask = token_loss.to("cuda:0") * decoder_mask_cut.to("cuda:0")
        batch_loss_sum_token = token_loss_with_mask.sum(dim=-1)
        batch_loss_mean_token = batch_loss_sum_token / decoder_lens.float().to("cuda:0")
        result_loss = batch_loss_mean_token.mean()

        #uii=1
        #if uii==1 :
        '''zuizhong=[]
        for jiji in range(final_dist.shape[0]):
        
                
                final_dist1=final_dist[jiji].reshape(1,-1)
                decoder_status1 = (decoder_status[0][:,0,:].reshape(1,1,-1),decoder_status[1][:,0,:].reshape(1,1,-1))
                context_vec1=context_vec[jiji].reshape(1,-1)
                next_coverage1=next_coverage[jiji].reshape(1,-1)
                coverage1=coverage[jiji].reshape(1,-1)
                beams = [Beam(tokens = [self.start_token_idx],log_probs = [1.0],status=decoder_status1,context_vec = context_vec1,coverage = coverage1)]

                
            
                log_probs = torch.log(final_dist1)
                #print(log_probs,log_probs.shape,"sasasa")
            # 最高2*10个概率的下标以及概率
                print(log_probs.shape,"log_probs")
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2,dim=-1)
                print(topk_ids,topk_log_probs,"topk_log_probstopk_log_probstopk_log_probs")
            # code.interact(local = locals())
                print(topk_ids.shape,"topk_ids")
                print(len(beams),"len(beams)")
                all_beams = []
                for i in range(len(beams)):
                    beam = beams[i] # 依次取第i个句子
                    h_i = decoder_status1[0][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                    c_i = decoder_status1[1][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                    status_i = (h_i,c_i)
                    context_vec_i = context_vec1[i,:].unsqueeze(0)   # keep dim (batch,hidden*2)
                    if self.use_coverage:
                        coverage_i = next_coverage1[i,:].unsqueeze(0)  # keep dim (batch,seq_len)
                    else:
                        coverage_i = None
                    #print(topk_ids,"topk_idstopk_idstopk_ids")
                    for j in range(beam_size*2):
                        print(topk_ids[i,j].item(),"句子数列")
                        
                        new_beam = beam.update(token=topk_ids[i,j].item(),
                                           log_prob = topk_log_probs[i,j].item(),
                                           status=status_i,
                                           context_vec = context_vec_i,
                                           coverage = coverage_i)
                        all_beams.append(new_beam)

                beams = []
                print(len(beam.tokens))
                for beam in sort_beams(all_beams):
                    if beam.tokens[-1] == self.stop_token_idx:
                        print("1111")
                        if len(beam.tokens) > self.min_decoder_len:
                            result.append(beam)
                        else:             # 如果以stop_token_idx 结尾，并且不够长，就丢弃掉，假如全部被丢掉了,0 == len(beams)
                            pass          # 把beam_size适当调大，min_decoder_len适当调小，如果还不行，模型估计废了。。。。。
                    else:
                        beams.append(beam)
                    if beam_size == len(beams) or len(result) == beam_size:
                        break
                step += 1

                if 0 == len(result):
                    result = beams
                
                sorted_result = sort_beams(result)
                zuizhong.append(sorted_result[0])'''
        #return result_loss,sorted_result[0]

        return result_loss
    def _eval(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,beam_size,reconstructed_unpacked,h):

        encoder_seq_len = encoder_mask.sum(dim = -1) # 计算一个batch中正文的长度
        max_encoder_len = encoder_seq_len.max()
        encoder_mask = encoder_mask[:,:max_encoder_len]
        encoder_with_oov = encoder_with_oov[:,:max_encoder_len]
        if self.use_coverage:
            coverage = coverage[:,:max_encoder_len]

        
        
        
        
        
        
        
        # encoder_outputs=[1,216,512] encoder_hidden[0].shape=[2,1,256] encoder_hidden[1].shape=[2,1,256]
        encoder_outputs1, encoder_hidden1 = self.encoder(encoder_input, encoder_mask)
        
        mmm=reconstructed_unpacked.shape[1]
        
        encoder_outputs2 = F.pad(reconstructed_unpacked, (0, 0, 0, 470 - mmm, 0, 0))
        encoder_hidden2 =h
        encoder_outputs=self.enco(torch.cat([encoder_outputs1, encoder_outputs2], dim=2 ))
        encoder_hidden=(self.hidd1(torch.cat([encoder_hidden1[0], encoder_hidden2[0]], dim=2 )),self.hidd2(torch.cat([encoder_hidden1[1], encoder_hidden2[1]], dim=2)))

         # decoder_status[0].shape=[1,1,256] decoder_status[1].shape=[1,1,256]

        # code.interact(local = locals())
        
        zuizhong=[]
        for jiji in range(encoder_input.shape[0]):
            


            
            encoder_hidden1 = (encoder_hidden[0][:,jiji,:].reshape(4,1,-1),encoder_hidden[1][:,jiji,:].reshape(4,1,-1))
            encoder_mask1=encoder_mask[jiji].reshape(1,-1)
            encoder_with_oov1=encoder_with_oov[jiji].reshape(1,-1)
            oovs_zero1=oovs_zero[jiji].reshape(1,-1)
            context_vec1=context_vec[jiji].reshape(1,-1)
            coverage1=coverage[jiji].reshape(1,-1)
            encoder_outputs1=encoder_outputs[jiji,:,:].reshape(1,470,-1)
            
            decoder_status = self.reduce(*encoder_hidden1)
            

            
            
            
            #beams = [Beam(tokens = [self.start_token_idx],log_probs = [1.0],status=decoder_status1,context_vec = context_vec1,coverage = coverage1)]
            
            
            # context_vec 其实是时间步step的记忆，含上文的语义，decoder_status是正文的上下文的语义
            beams = [Beam(tokens = [self.start_token_idx],log_probs = [1.0],status=decoder_status,context_vec = context_vec1,
                      coverage = coverage1)]

            step = 0
            result = []

            while step < self.max_decoder_len and len(result) < 4:
                current_tokens_idx = [b.tokens[-1] for b in beams] # 取出时间步上当前的step的 idx 值
                #print(current_tokens_idx,"current_tokens_idx")
                current_tokens_idx = [token_idx if token_idx < self.vob_size else self.unk_token_idx
                                      for token_idx in current_tokens_idx]
                #print(current_tokens_idx,"current_tokens_idxcurrent_tokens_idx")

                decoder_input_one_step = torch.tensor(current_tokens_idx,dtype=torch.long)
                
                

                decoder_input_one_step = decoder_input_one_step.to(encoder_outputs1.device)

                # 取出上下文语义的隐状态
                status_h_list = [b.status[0] for b in beams]
                status_c_list = [b.status[1] for b in beams]

                decoder_h = torch.cat(status_h_list,dim=1)   # status_h  (num_layers * num_directions, batch, hidden_size)
                decoder_c = torch.cat(status_c_list,dim=1)   # status_c  (num_layers * num_directions, batch, hidden_size)
                decoder_status = (decoder_h,decoder_c)

                context_vec_list = [b.context_vec for b in beams]
                context_vec1 = torch.cat(context_vec_list,dim=0)     # context_vec (batch,hidden*2)

                if self.use_coverage:
                    coverage_list = [b.coverage for b in beams]
                    coverage1 = torch.cat(coverage_list,dim=0)                 # coverage (batch,seq_len)
                else:
                    coverage1 = None

                if 1 == step:
                    encoder_outputs1 = encoder_outputs1.expand(beam_size,encoder_outputs1.size(1),encoder_outputs1.size(2))
                    encoder_mask1 = encoder_mask1.expand(beam_size,encoder_mask1.shape[1])
                    if oovs_zero1 is not None:
                        oovs_zero1 = oovs_zero1.expand(beam_size, oovs_zero1.shape[1])
                    encoder_with_oov1 = encoder_with_oov1.expand(beam_size, encoder_with_oov1.shape[1])


                

                final_dist, decoder_status, context_vec1, attention_score, genera_p, next_coverage = \
                    self.decoder(
                    decoder_input_one_step=decoder_input_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs1,
                    encoder_mask=encoder_mask1,
                    context_vec=context_vec1,
                    oovs_zero=oovs_zero1,
                    encoder_with_oov=encoder_with_oov1,
                    coverage=coverage1,
                    step=step)

                # (batch_size,vob_size)
                log_probs = torch.log(final_dist)
                # 最高2*10个概率的下标以及概率
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2,dim=-1)
                # code.interact(local = locals())

                all_beams = []
                for i in range(len(beams)):
                    beam = beams[i] # 依次取第i个句子
                    h_i = decoder_status[0][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                    c_i = decoder_status[1][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                    status_i = (h_i,c_i)
                    context_vec_i = context_vec1[i,:].unsqueeze(0)   # keep dim (batch,hidden*2)
                    if self.use_coverage:
                        coverage_i = next_coverage[i,:].unsqueeze(0)  # keep dim (batch,seq_len)
                    else:
                        coverage_i = None

                    for j in range(beam_size*2):
                        new_beam = beam.update(token=topk_ids[i,j].item(),
                                           log_prob = topk_log_probs[i,j].item(),
                                           status=status_i,
                                           context_vec = context_vec_i,
                                           coverage = coverage_i)
                        all_beams.append(new_beam)

                beams = []
                for beam in sort_beams(all_beams):
                    if beam.tokens[-1] == self.stop_token_idx:
                        if len(beam.tokens) > self.min_decoder_len:
                            result.append(beam)
                        else:             # 如果以stop_token_idx 结尾，并且不够长，就丢弃掉，假如全部被丢掉了,0 == len(beams)
                            pass          # 把beam_size适当调大，min_decoder_len适当调小，如果还不行，模型估计废了。。。。。
                    else:
                        beams.append(beam)
                    if beam_size == len(beams) or len(result) == beam_size:
                        break
                step += 1

            if 0 == len(result):
                result = beams

            sorted_result = sort_beams(result)
            zuizhong.append(sorted_result[0])
        return zuizhong
    
    def _decoder(self,encoder_input,encoder_mask,encoder_with_oov,oovs_zero,context_vec,coverage,beam_size = 4):

        encoder_seq_len = encoder_mask.sum(dim = -1) # 计算一个batch中正文的长度
        max_encoder_len = encoder_seq_len.max()
        encoder_mask = encoder_mask[:,:max_encoder_len]
        encoder_with_oov = encoder_with_oov[:,:max_encoder_len]
        if self.use_coverage:
            coverage = coverage[:,:max_encoder_len]

        # encoder_outputs=[1,216,512] encoder_hidden[0].shape=[2,1,256] encoder_hidden[1].shape=[2,1,256]
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)
        decoder_status = self.reduce(*encoder_hidden) # decoder_status[0].shape=[1,1,256] decoder_status[1].shape=[1,1,256]

        # code.interact(local = locals())

        # context_vec 其实是时间步step的记忆，含上文的语义，decoder_status是正文的上下文的语义
        beams = [Beam(tokens = [self.start_token_idx],log_probs = [1.0],status=decoder_status,context_vec = context_vec,
                      coverage = coverage)]

        step = 0
        result = []

        while step < self.max_decoder_len and len(result) < 4:
            current_tokens_idx = [b.tokens[-1] for b in beams] # 取出时间步上当前的step的 idx 值
            current_tokens_idx = [token_idx if token_idx < self.vob_size else self.unk_token_idx
                                      for token_idx in current_tokens_idx]

            decoder_input_one_step = torch.tensor(current_tokens_idx,dtype=torch.long)

            decoder_input_one_step = decoder_input_one_step.to(encoder_outputs.device)

            # 取出上下文语义的隐状态
            status_h_list = [b.status[0] for b in beams]
            status_c_list = [b.status[1] for b in beams]

            decoder_h = torch.cat(status_h_list,dim=1)   # status_h  (num_layers * num_directions, batch, hidden_size)
            decoder_c = torch.cat(status_c_list,dim=1)   # status_c  (num_layers * num_directions, batch, hidden_size)
            decoder_status = (decoder_h,decoder_c)

            context_vec_list = [b.context_vec for b in beams]
            context_vec = torch.cat(context_vec_list,dim=0)     # context_vec (batch,hidden*2)

            if self.use_coverage:
                coverage_list = [b.coverage for b in beams]
                coverage = torch.cat(coverage_list,dim=0)                 # coverage (batch,seq_len)
            else:
                coverage = None

            if 1 == step:
                encoder_outputs = encoder_outputs.expand(beam_size,encoder_outputs.size(1),encoder_outputs.size(2))
                encoder_mask = encoder_mask.expand(beam_size,encoder_mask.shape[1])
                if oovs_zero is not None:
                    oovs_zero = oovs_zero.expand(beam_size, oovs_zero.shape[1])
                encoder_with_oov = encoder_with_oov.expand(beam_size, encoder_with_oov.shape[1])



            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_input_one_step=decoder_input_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)

            # (batch_size,vob_size)
            log_probs = torch.log(final_dist)
            # 最高2*10个概率的下标以及概率
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2,dim=-1)
            # code.interact(local = locals())

            all_beams = []
            for i in range(len(beams)):
                beam = beams[i] # 依次取第i个句子
                h_i = decoder_status[0][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                c_i = decoder_status[1][:,i,:].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                status_i = (h_i,c_i)
                context_vec_i = context_vec[i,:].unsqueeze(0)   # keep dim (batch,hidden*2)
                if self.use_coverage:
                    coverage_i = next_coverage[i,:].unsqueeze(0)  # keep dim (batch,seq_len)
                else:
                    coverage_i = None

                for j in range(beam_size*2):
                    new_beam = beam.update(token=topk_ids[i,j].item(),
                                           log_prob = topk_log_probs[i,j].item(),
                                           status=status_i,
                                           context_vec = context_vec_i,
                                           coverage = coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for beam in sort_beams(all_beams):
                if beam.tokens[-1] == self.stop_token_idx:
                    if len(beam.tokens) > self.min_decoder_len:
                        result.append(beam)
                    else:             # 如果以stop_token_idx 结尾，并且不够长，就丢弃掉，假如全部被丢掉了,0 == len(beams)
                        pass          # 把beam_size适当调大，min_decoder_len适当调小，如果还不行，模型估计废了。。。。。
                else:
                    beams.append(beam)
                if beam_size == len(beams) or len(result) == beam_size:
                    break
            step += 1

        if 0 == len(result):
            result = beams

        sorted_result = sort_beams(result)

        return sorted_result[0]





































































