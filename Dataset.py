import torch 
from torch import nn 
import numpy as np
import os,sys,re
from collections import Counter
import nltk
nltk.load('english', format='text')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class FairytalesDataset(torch.utils.data.Dataset): 
    
    def __init__(self, dir_path, start_token, end_token, args): 
        self.args = args
        self.start_token = start_token
        self.end_token = end_token
        self.stop_words = stopwords.words('english') 
        self.dir_path = dir_path
        
        self.words = self.load_words()
        self.unique_words = self.get_unique_words()
        
        # {index: word}
        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        # {word: index}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]
        
    def tokens_to_text(self, st_text, stop_words, use_stop_words): 
        l=[]
        for s in st_text: # loop through each sentence
            # keep alphanumeric
            if use_stop_words:
                s = [x for x in s if x.isalnum()]
            else:
                s = [x for x in s if x.isalnum() and x not in stop_words]
            s.insert(0, self.start_token) # insert start/end tokens
            s.append(self.end_token)
            l.extend(s)  
            
        return l
    
    def load_words(self): 
        with open ("data/fairytales.txt","r", encoding="utf-8") as f:
             fairytales_text = f.read()
        # get rid of newline symbols, tokenize text to sentences and words, lowercase the text
        # add BOS and EOS tokens
        st_fairytales_text = sent_tokenize(fairytales_text.lower())
        st_fairytales_text = [word_tokenize(x) for x in st_fairytales_text]

        word_fairytales_text_with_sw = self.tokens_to_text(st_fairytales_text,
                                              self.stop_words,
                                              use_stop_words=True)

        word_fairytales_text = word_fairytales_text_with_sw
        
        return word_fairytales_text
    
    def get_unique_words(self): 
        word_count = Counter(self.words)
        return sorted(word_count, key=word_count.get, reverse=True)
        
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
    
    def __getitem__(self, index): 
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )
        