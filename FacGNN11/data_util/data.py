import en_config as config
import torch
import os
import logging
import code
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class Feature(object):
    def __init__(self,article,abstract,unique_name,encoder_input,decoder_input,decoder_target,encoder_input_with_oov,oovs,
                 decoder_target_with_oov,max_encoder_len,max_decoder_len,pad_idx = 0):

        assert len(decoder_input) == len(decoder_target)
        self.article = article
        self.abstract = abstract

        self.unique_name = unique_name
        self.encoder_input,self.encoder_mask = self._add_pad_and_gene_mask(encoder_input,max_encoder_len,pad_idx)
        self.encoder_input_with_oov = self._add_pad_and_gene_mask(encoder_input_with_oov, max_encoder_len, pad_idx,
                                                                  return_mask=False)

        self.decoder_input,self.decoder_mask = self._add_pad_and_gene_mask(decoder_input,max_decoder_len,pad_idx)

        self.decoder_target = self._add_pad_and_gene_mask(decoder_target,max_decoder_len,pad_idx,return_mask=False)
        self.decoder_target_with_oov = self._add_pad_and_gene_mask(decoder_target_with_oov, max_decoder_len, pad_idx,
                                                                   return_mask=False)
        self.oovs = oovs
        self.oov_len = len(oovs)

    @classmethod
    def _add_pad_and_gene_mask(cls,x,max_len,pad_idx = 0,return_mask = True):
        pad_len = max_len - len(x)
        assert pad_len >= 0

        if return_mask:
            mask = [1] * len(x)
            mask.extend([0] * pad_len)
            assert len(mask) == max_len

        x.extend([pad_idx] * pad_len)
        assert len(x) == max_len

        if return_mask:
            return x,mask
        else:
            return x


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line:
      return line
  if line=="":
      return line
  if line[-1] in config.END_TOKENS:
      return line
  return line + " ."


def article_word_to_idx_with_oov(article_list,vocab):
    indexes = []
    oovs = []
    for word in article_list:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word not in oovs:
                oovs.append(word)
            oov_idx = oovs.index(word)
            indexes.append(vocab.get_vob_size() + oov_idx)
        else:
            indexes.append(idx)
    return indexes,oovs


def abstract_target_idx_with_oov(abstract_list , vocab , oovs):
    target_with_oov = []
    for word in abstract_list[1:]:
        idx = vocab.word_2_idx(word)
        if vocab.unk_idx == idx:
            if word in oovs:
                target_with_oov.append(vocab.vob_num+oovs.index(word))
            else:
                target_with_oov.append(vocab.unk_idx)
        else:
            target_with_oov.append(idx)
    return target_with_oov


def read_example_convert_to_feature(text,abstr,article_len,abstract_len,vocab,index,point = True):
    
    #article = []
    abstract = []
    flag = False
    
    #example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
 
    abstract=" ".join(abstr).split(' ')
    if abstract[0]=='–':
        abstract=abstract[1:]
    article=" ".join(sum(text,[])).split(' ')
    #abstract.extend(line.split(' '))

     
    
    #article.extend(line.split(' '))
    #f.close()
    
    '''for ii in e["label"]:
        
        abstract.extend([e["text"][ii]])
        
    print(abstract)
    abstract=" ".join(abstract).split(' ')'''


    # 有一些没有摘要，有一些没有文章，都不要了
    if 0 == len(article) or 0 == len(abstract):
        return None
    #unique_name = example_path_name.split("\\")[-1].split(".")[0]
    unique_name = index
    print_idx = 20
    #if index < print_idx:
    #    print("====================================={}=====================================".format(unique_name))


    
    #print("原始文章长度[{}]===[{}]".format(len(article),article))
    #print("原始摘要长度[{}]===[{}]".format(len(abstract),abstract))
    
    # 截断,限制文章的长度
    article = article[:article_len] # 原文长度840 ，截断400

    
    #print("截断后的文章长度[{}]===[{}]".format(len(article),(article)))
    article_indexes = [vocab.word_2_idx(word) for word in article] # 将正文转化为向量
    
    


    #加上 start 和 end
    abstract = [vocab.start_token] + abstract + [vocab.stop_token]
    #print(abstract,"abstract")
    # 截断，限制摘要的长度
    abstract = abstract[:abstract_len+1] # 摘要加上start 和 end token之后再截断
    #print(abstract)
    
    #print("截断后的摘要长度[{}]===[{}]".format(len(abstract),abstract))



    abstract_indexes = [vocab.word_2_idx(word) for word in abstract] # 将摘要转化为向量

    decoder_input = abstract_indexes[:-1] # 构建输入的摘要和目标摘要，语言模型
    decoder_target = abstract_indexes[1:]

    assert len(decoder_input) == len(decoder_target)

    if point:
        # 更新正文和摘要的词表，
        encoder_input_with_oov,oovs = article_word_to_idx_with_oov(article,vocab)
        decoder_target_with_oov = abstract_target_idx_with_oov(abstract,vocab,oovs)


    feature_obj = Feature(article = article,
                          abstract = abstract[1:],
                          unique_name = unique_name,
                          encoder_input =article_indexes,
                          decoder_input = decoder_input,
                          decoder_target = decoder_target,
                          encoder_input_with_oov = encoder_input_with_oov,
                          oovs = oovs,
                          decoder_target_with_oov = decoder_target_with_oov,
                          max_encoder_len = article_len,
                          max_decoder_len = abstract_len,
                          pad_idx = vocab.pad_idx)
    # code.interact(local = locals())


    

    '''print("encoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input])))
    print("encoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_mask])))
    print("encoder_input_with_oov :[{}]".format(" ".join([str(i) for i in feature_obj.encoder_input_with_oov])))
    print("decoder_input :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_input])))
    print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_mask])))
    print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target])))
    print("decoder_mask  :[{}]".format(" ".join([str(i) for i in feature_obj.decoder_target_with_oov])))
    print("oovs          :[{}]".format(" ".join(oovs)))
    print("\n")'''

    return feature_obj


def get_features(text,abstr,idx,vocab,data_set = "train",example_num = 1024*8):
    

    assert data_set in ["train","test","val"]
    assert 0 == example_num % 1024 and example_num > 1024

    
    #token_file_list = os.listdir(token_dir)
    #print(token_dir,"token_dir")
    #sample_num = len(token_file_list) # 287227

    
    feature_file_prefix = "{}".format(data_set) # 'train'
    features = []
 
    

    
            
    feature_obj = read_example_convert_to_feature(text=text,abstr=abstr,article_len=470,abstract_len = 170,index=idx,vocab = vocab)
            
    #train_dataset = get_features_from_cache(feature_obj)
    #batch = from_feature_get_model_input(train_dataset,hidden_dim=256,device = 'cpu',pointer_gen = True,use_coverage = True)
            

    return feature_obj 
        

        
def get_features_from_cache(cache_file):

    f = cache_file

    all_encoder_input = torch.tensor([f.encoder_input ], dtype=torch.long)
    all_encoder_mask = torch.tensor([f.encoder_mask ],dtype = torch.long)

    all_decoder_input = torch.tensor([f.decoder_input ],dtype=torch.long)
    all_decoder_mask = torch.tensor([f.decoder_mask ],dtype=torch.int)

    all_decoder_target = torch.tensor([f.decoder_target ],dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor([f.encoder_input_with_oov ],dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor([f.decoder_target_with_oov ],dtype=torch.long )
    all_oov_len = torch.tensor([f.oov_len ],dtype=torch.int)

    dataset = [all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                            all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len]


    
    return dataset

def from_feature_get_model_input(features,hidden_dim,device = torch.device("cpu"),pointer_gen = True,
                                 use_coverage = True):

    all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len = features
    
   
    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if pointer_gen:                # 当时用指针网络时，decoder_target应该要带上oovs
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:                # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
            oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32)
    else:                                  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
        all_encoder_input_with_oov = None


    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(),dtype=torch.float32)          # 注意数据格式是float

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)   # 注意数据格式是float

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
    model_input = [t.to(device) if t is not None else None for t in model_input]

    return model_input