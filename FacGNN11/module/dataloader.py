
from data_util.data import get_features
from gensim import corpora, models, similarities
import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *

import dgl
from dgl.data.utils import save_graphs, load_graphs

from collections import Counter
import ast

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/','!',',','.','?','-s','-ly','</s>','s','(',')','’','.','i','.i',':','','"']
FILTERWORD.extend(punctuations)




class Example(object):


    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):


        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        # label_shape = (len(self.original_article_sents), len(self.original_article_sents))
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):

        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Example2(Example):


    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):


        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


######################################### ExampleSet #########################################

class ExampleSet(torch.utils.data.Dataset):


    def __init__(self, data_path,entity_path, vocab,vocab2, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path):
     
        path=entity_path
        self.entity_text= readText(path)
        
        self.vocab = vocab
        self.vocab2 = vocab2
        self.sent_max_len = sent_max_len  #限制输入句子的最大长度###
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path) #通过地址读取到文本### readjson函数 with open(fname, encoding="utf-8") as f: 读取一行 一行即是一篇文章###
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)#读取预处理好的词频文件###
        self.filterwords = FILTERWORD#对数据进行清洗，过滤掉常用词已经标点符号###
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))

                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)

    def get_example(self, index):   #取出文件中的 文章 人工总结 以及标签
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example,e["text"],e["summary"]

    def pad_label_m(self, label_matrix):    #统一句子节点长度
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):  #添加词节点
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys(): #判断这个词 是否为标点或者是 常用词   并且判断是否已
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)  #统计 词的个数

        G.add_nodes(w_nodes)  #通过个数确定要加添的 词节点个数
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes) 
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values())) #词节点在 词表中的ID
        G.ndata["dtype"] = torch.zeros(w_nodes) #表明 它是词节点   unit=0 方便查询

        return wid2nid, nid2wid

    def CreateGraph(self, input_pad, label, w2s_w,entity): #构建图结构
        entity = entity.replace('nan', '0.01')
        entity_list = ast.literal_eval(entity)


            

            
        e_data = {}
        for item, value in entity_list[0]:
            phrase, index = item
            if phrase in self.filterwords:
                continue
            if phrase in e_data:
                e_data[phrase][0].append(index)
            else:
                e_data[phrase] = [[index], value]
        e_list = [([phrase, *indexes], value) for phrase, (indexes, value) in e_data.items()]
        
        p_data = {}
        for item, value in entity_list[1]:
            phrase, index = item
            if phrase in self.filterwords:
                continue
            if phrase in p_data:
                p_data[phrase][0].append(index)
            else:
                p_data[phrase] = [[index], value]
        p_list = [([phrase, *indexes], value) for phrase, (indexes, value) in p_data.items()]
        ep_list= e_list+p_list
        if len(ep_list)==0:
            ep_list=[(['you', 1], 0.01)]
        
        
        
        G = dgl.DGLGraph() #首先获取一副空图
        #G = dgl.graph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)# 调用词节点添加程序  返回值为  文章中包含哪些词节点  以及它们的编号
        w_nodes = len(nid2wid)
        #print(w_nodes)
        N = len(input_pad)  #句子个数

        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)] #句子节点编号 在词节点之后
        #print(sentid2nid,"fsfcsfcsfcsfsf")
        
        Nnum=len(ep_list)
        
        G.add_nodes(Nnum)#添加实体节点，标记为2
        abc=N+w_nodes
        G.ndata["dtype"][abc:] = torch.tensor([2.]*Nnum)
        G.ndata["unit"][abc:] = torch.tensor([2.]*Nnum)

        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N): #以迭代的方式将句子与词节点间的 词频/出度逆函数值 添加到它们的边连接中
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            
            sent_tfw = w2s_w[str(i)]

            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})


            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
            
        
        
        '''for idi in snode_id:
            G.add_edges(idi, snode_id1, data={"dtype": torch.Tensor([2])})
            G.add_edges(snode_id1, idi,  data={"dtype": torch.Tensor([2])}) '''   
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]
        enid = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        #G.nodes[sentid2nid].data["shu"] = file_path_name
        
        
        
        entity_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        
        cc=[]
        jj=[]
        #dianwei=0
        if len(entity_id)!=0 :
            for  bot_entity in ep_list :
                entiy_input=[]
                root_in=[]
                article_words = bot_entity[0][0].split()
                root_in=bot_entity[1]
                entiy_input.extend([self.vocab.word2id(w.lower()) for w in article_words])
                if len(entiy_input) < 100 :
                    entiy_input.extend([0] * (100 - len(entiy_input)))
                if len(entiy_input) > 100 :
                    entiy_input = entiy_input[:100]
                cc.append(entiy_input)
                jj.append(root_in)
        
        


            
        try:
            if len(entity_id)!=0 :
                G.nodes[entity_id].data["root"]=torch.tensor(jj)
                G.nodes[entity_id].data["words"]=torch.LongTensor(cc)
                G.nodes[entity_id].data["position"] = torch.arange(1, Nnum+1).view(-1, 1).long()  
        except Exception as e:
            print(len(cc),"cc")
            print(len(entity_id),"entity_id")
            print(entity_list)
        #print(torch.arange(1, len(entity_id)+1).view(-1, 1).long() ,len(entity_id))    
            
        dianwei=0
        if len(entity_id)!=0 :
            for  bot_entity in ep_list :
                list_lis =Counter(bot_entity[0][1:])
                for k,v in list_lis.items():
                    try:  
                        G.add_edges(w_nodes+int(k), w_nodes+N+dianwei, data={"dtype": torch.Tensor([2]),"feat":torch.LongTensor([v])})
                        G.add_edges(w_nodes+N+dianwei, w_nodes+int(k), data={"dtype": torch.Tensor([2]),"feat":torch.LongTensor([v])})
                    except:
                        print(w_nodes+int(k),"juzi",w_nodes+N+dianwei,"en")
                        print(entity_id,"en")
                        print(sentid2nid,"sen")
                    
                dianwei=dianwei+1
        

        

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        
        item,text,abtr = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]
        

        file_path_name=get_features(text=text,abstr=abtr,idx=index,vocab = self.vocab2,data_set="train")   
        
        G = self.CreateGraph(input_pad, label, w2s_w, self.entity_text[index])
        
        
        
         



        return G, index,file_path_name

    def __len__(self):
        return self.size


class MultiExampleSet(ExampleSet):
    """ Constructor: Dataset of example(object) for multiple document summarization"""
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, w2d_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param w2d_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2dTFIDF.py)
        """

        super().__init__(data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path)

        logger.info("[INFO] Loading word2doc TFIDF file from %s!" % w2d_path)
        self.w2d_tfidf = readJson(w2d_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1
                if sentNo >= sentNum:
                    return sent2doc
        return sent2doc

    def CreateGraph(self, docLen, sent_pad, doc_pad, label, w2s_w, w2d_w):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
        """
        # add word nodes
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        N = len(sent_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        ws_nodes = w_nodes + N

        # add doc nodes
        sent2doc = self.MapSent2Doc(docLen, N)
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        docid2nid = [i + ws_nodes for i in range(article_num)]

        # add sent edges
        for i in range(N):
            c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            try:
                sent_tfw = w2s_w[str(i)]
            except:
                print("1")
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    # w2s s2w
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            G.add_edge(sent_nid, docnid, data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i])
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        enc_doc_input = item.enc_doc_input
        article_len = item.article_len
        label = self.pad_label_m(item.label_matrix)

        G = self.CreateGraph(article_len, sent_pad, enc_doc_input, label, self.w2s_tfidf[index], self.w2d_tfidf[index])

        return G, index


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


######################################### Tools #########################################


import dgl


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def graph_collate_fn(samples):
    

    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index,f = map(list, zip(*samples))
    

    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph

    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    #print(len(sorted_index),"sort")
   #try:
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    #except:
        #print(sorted_index,"saa",len(sorted_index))

    train_dataset,oovs,abstracts,article = get_features_from_cache(f)
    
    batch = from_feature_get_model_input(train_dataset,hidden_dim=256,device = 'cpu',pointer_gen = True,use_coverage = True)
            
    return batched_graph, [index[idx] for idx in sorted_index],batch,oovs,abstracts,article










def get_features_from_cache(cache_file):

    features = cache_file

    all_encoder_input = torch.tensor([f.encoder_input for f in features], dtype=torch.long)
    
    all_encoder_mask = torch.tensor([f.encoder_mask for f in features],dtype = torch.long)

    all_decoder_input = torch.tensor([f.decoder_input for f in features],dtype=torch.long)
    all_decoder_mask = torch.tensor([f.decoder_mask for f in features],dtype=torch.int)

    all_decoder_target = torch.tensor([f.decoder_target for f in features],dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor([f.encoder_input_with_oov for f in features],dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor([f.decoder_target_with_oov for f in features],dtype=torch.long )
    all_oov_len = torch.tensor([f.oov_len for f in features],dtype=torch.int)
    
    

    dataset = [all_encoder_input, all_encoder_mask, all_decoder_input,all_decoder_mask,all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len]
    
    abstracts = [f.abstract for f in features]
    oovs = [f.oovs for f in features]

    article = [f.article for f in features]
    return dataset,oovs,abstracts,article
    
    
    
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

    init_context_vec = torch.zeros((batch_size,  hidden_dim),dtype=torch.float32)   # 注意数据格式是float

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
    model_input = [t.to(device) if t is not None else None for t in model_input]

    return model_input