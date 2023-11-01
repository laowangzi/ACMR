from utils import *
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from models import ASCMR
import os
import sys
import warnings
import gc
from torch.optim.lr_scheduler import StepLR
warnings.filterwarnings('ignore')
config = {}
# 'ca', 'fr', 'in', 'jp', 'mx', 'uk', 'de'
all_mkt = ['de', 'jp', 'ca', 'fr', 'in', 'mx', 'uk']
config['neg_num'] = 4

config['patience'] = 3
config['latent_dim'] = 32
config['dropout'] = 0.5
config['lr'] = 1e-3
config['l2_reg'] = 1e-8
config['num_epoch'] = 50

config['device'] = 'cuda:0'
config['dnn_use_bn'] = False
config['l2_reg_dnn'] = 0
config['batch_size'] = 2048
config['layers'] = [config['latent_dim']*2, config['latent_dim']*4, config['latent_dim']*2, config['latent_dim']]
#bert
config['bert_max_len'] = 51
config['bert_num_blocks'] = 4
config['bert_num_heads'] = 4

cold_start = None

for current_mkt in all_mkt:
    config['tgt_mkt'] = [current_mkt]
    config['src_mkt'] = list(set(all_mkt)-set([current_mkt]))
    print('target market:', current_mkt)
    print('source market:', config['src_mkt'])
    config['save_path'] = f'./checkpoint/ASCMR_{current_mkt}_pretrain.pt'
    config['num_mkts'] = len(set(config['tgt_mkt']+config['src_mkt']))
    id_bank = ID_Bank()
    generator = DataGenerator(src_markets=config['src_mkt'], target_market=config['tgt_mkt'],id_bank=id_bank, neg_num=config['neg_num'])
    config['user_num'] = id_bank.last_user_index+1
    config['item_num'] = id_bank.last_item_index+1
    train, valid, test = generator.generate_data()



    train_finetune = train[train['market'] == generator.mkt_dict[config['tgt_mkt'][0]]]


    loader_generator = ASCMR_loader(train, valid, test, config)
    train_loader = loader_generator.get_loader('train')


    model = ASCMR(config)
    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, config)
    model = model.to(config['device'])


    best_score  = 0.0
    patience = config['patience']
    for epoch in range(config['num_epoch']):
        model.train()
        start = time.time()
        print(f'epoch {epoch+1} start!')
        for data in train_loader:
            optimizer.zero_grad()
            x, mkt, his_num, his_mask, target = data
            x, mkt, his_num, his_mask, target = x.to(config['device']), mkt.to(config['device']), his_num.to(config['device'])                                        ,his_mask.to(config['device']), target.to(config['device'])
            ratings = model(x, mkt, his_num, his_mask)
            loss = criterion(ratings.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
        result = eval_model_ASCMR(model, valid, config, 'valid', loader_generator)
        if result['ndcg@10'] > best_score:
            best_score = result['ndcg@10']
            save(model, config)
        else:
            patience -= 1
        if patience <= 0:
            break
        end = time.time()
        print(f'train time {round(end-start)}s')
        print('-'*90)


    model = load_model(model, config)
    res = eval_model_ASCMR(model, test, config, 'test', loader_generator)

    #fine-tuning stage
    loader_generator = Bert4XMR_loader(train_finetune, valid, test, config)
    train_loader = loader_generator.get_loader('train')
    best_score  = 0.0
    patience = config['patience']
    for epoch in range(config['num_epoch']):
        model.train()
        start = time.time()
        print(f'epoch {epoch+1} start!')
        for data in train_loader:
            optimizer.zero_grad()
            x, mkt, his_num, his_mask, target = data
            x, mkt, his_num, his_mask, target = x.to(config['device']), mkt.to(config['device']), his_num.to(config['device'])                                        ,his_mask.to(config['device']), target.to(config['device'])
            ratings = model(x, mkt, his_num, his_mask)
            loss = criterion(ratings.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
        result = eval_model_ASCMR(model, valid, config, 'valid', loader_generator)
        if result['ndcg@10'] > best_score:
            best_score = result['ndcg@10']
            save(model, config)
        else:
            patience -= 1
        if patience <= 0:
            break
        end = time.time()
        print(f'train time {round(end-start)}s')
        print('-'*90)
    model = load_model(model, config)
    res = eval_model_ASCMR(model, test, config, 'test', loader_generator, cold_start=cold_start)

