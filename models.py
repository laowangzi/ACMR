import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import *
import time

class ACMR(nn.Module):
    def __init__(self, config):
        super(ACMR, self).__init__()
        self.config = config
        self.bert = BERT(config)
        self.out = nn.Linear(2*self.bert.hidden, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, market, history_num, sequenceMask):
        '''
        x: bs*seqlen
        market: bs*seqlen
        history_num: bs
        sequenceMask: bs*(seqlen-1)
        '''
        x = self.bert(x, market)
        
        userMatrix = x[:, :-1, :]    #bs*seqlen*hiden
        #mean pooling
        user = torch.matmul(sequenceMask.unsqueeze(dim=1), userMatrix).squeeze() / history_num.unsqueeze(dim=1) #bs*hidden
        item = x[:,-1,:].squeeze()  # batch*hidden        
        
        x = self.out(torch.cat((user,item) , -1))
        x = self.activation(x)
        return x
    
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        max_len = config['bert_max_len']
        num_users = int(config['user_num']+1)
        num_items = int(config['item_num']+1)
        n_layers = config['bert_num_blocks']
        heads = config['bert_num_heads']
        vocab_size = num_items + 2
        hidden = config['latent_dim']
        self.hidden = hidden
        dropout = config['dropout']
        num_markets = config['num_mkts']

        # embedding for BERT
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, num_markets=num_markets, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden,nhead=heads, dim_feedforward=hidden * 4,batch_first=True, dropout=dropout)
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)      

    def forward(self, x, markets):
        mask = (x <= 0)
        x = self.embedding(x, markets)
        x = self.transformer_blocks(x, src_key_padding_mask=mask)
        return x

class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, max_len, num_markets, dropout=0.3):

        super(BERTEmbedding, self).__init__()
        self.token = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size, padding_idx=0)
        #market embedding
        self.market_embedding = nn.Embedding(num_markets+1 , embed_size)
        self.LayerNorm = nn.LayerNorm(embed_size, eps = 1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        
#         nn.init.xavier_uniform_(self.token.weight)
#         print('xavier_uniform_')

    def forward(self, sequence, mkts):
        #item embs + mkt embs
        x = self.token(sequence)+self.market_embedding(mkts)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x
