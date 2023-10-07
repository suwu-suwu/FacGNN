import os
import json
import en_config as config


PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

class Vocab2(object):
    def __init__(self,vocab_file,vob_num = 50000):

        # 关于路径，在win下，有时候反斜线会变成转义字符，导致路径不存在
        assert os.path.isfile(vocab_file)
        self.vob_num = vob_num
        
        self._count = 0
        
        self.word_to_idx = {}
        self.idx_to_word = {}

        '''for idx,token in enumerate(config.SPECIAL_TOKEN):
            self.word_to_idx[token] = idx
            self.idx_to_word.append(token)'''

        self.pad_idx = 0
        self.pad_token = config.SPECIAL_TOKEN[self.pad_idx]

        self.unk_idx = 1
        self.unk_token = config.SPECIAL_TOKEN[self.unk_idx]

        self.start_idx = 2
        self.start_token = config.SPECIAL_TOKEN[self.start_idx]

        self.stop_idx = 3
        self.stop_token = config.SPECIAL_TOKEN[self.stop_idx]

        for w in [PAD_TOKEN, UNKNOWN_TOKEN,  START_DECODING, STOP_DECODING]:
            self.word_to_idx[w] = self._count
            self.idx_to_word[self._count] = w
            self._count += 1

        with open(vocab_file,"r",encoding='utf-8') as f:
            #word_freq = json.load(f)
            word_freq = f
        

            special_len = len(self.idx_to_word)

            for token in f:
                pieces = token.split("\t")
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word_to_idx:
                    
                    continue

                #idx = special_len + i
                #if idx >= self.vob_num:
                #    break
                self.word_to_idx[w] = self._count
                self.idx_to_word[self._count] = w
                self._count += 1
                if self.vob_num != 0 and self._count >= self.vob_num:
                    
                    break
            f.close()

    def word_2_idx(self,word):
        return self.word_to_idx.get(word,self.unk_idx)



    def idx_2_word(self,idx):
        if (idx >= 0) and (idx < self.vob_num):
            return self.idx_to_word[idx]
        else:
            return self.unk_idx


    def get_vob_size(self):
        return self.vob_num













