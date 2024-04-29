from utils import *
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from models import ACMR
import os
import sys
import warnings
from config import config
warnings.filterwarnings('ignore')


def run(config):
    current_mkt = config['tgt_mkt'][0]
    print(config)
    print('target market:', current_mkt)
    print('source market:', config['src_mkt'])
    config['save_path'] = f'./checkpoint/ACMR_{current_mkt}_pretrain.pt'
    config['num_mkts'] = len(set(config['tgt_mkt']+config['src_mkt']))
    id_bank = ID_Bank()
    generator = DataGenerator(src_markets=config['src_mkt'], target_market=config['tgt_mkt'],id_bank=id_bank, neg_num=config['neg_num'])
    config['user_num'] = id_bank.last_user_index+1
    config['item_num'] = id_bank.last_item_index+1
    train, valid, test = generator.generate_data()

    train_finetune = train[train['market'] == generator.mkt_dict[config['tgt_mkt'][0]]]


    loader_generator = ACMR_loader(train, valid, test, config)
    train_loader = loader_generator.get_loader('train')


    model = ACMR(config)
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
        result = eval_model_ACMR(model, valid, config, 'valid', loader_generator)
        score = result['ndcg@5']
        if score > best_score:
            best_score = score
            save(model, config)
#             patience = config['patience']
        else:
            patience -= 1
        if patience <= 0:
            break
        end = time.time()
        print(f'train time {round(end-start)}s')
        print('-'*90)

    #load pretrained model
    model = load_model(model, config)
    res = eval_model_ACMR(model, test, config, 'test', loader_generator)

    #fine-tuning stage
    config['save_path'] = f'./checkpoint/ACMR_{current_mkt}_finetune.pt'
    loader_generator = ACMR_loader(train_finetune, valid, test, config)
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
        result = eval_model_ACMR(model, valid, config, 'valid', loader_generator)
        score = result['ndcg@5']
        if score > best_score:
            best_score = score
            save(model, config)
#             patience = config['patience']
        else:
            patience -= 1
        if patience <= 0:
            break
        end = time.time()
        print(f'train time {round(end-start)}s')
        print('-'*90)
    #test finetuned model
    model = load_model(model, config)
    res = eval_model_ACMR(model, test, config, 'test', loader_generator)

if __name__ == '__main__':
    run(config)
