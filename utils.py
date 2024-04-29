import sys
import math
import os
import random
import pandas as pd
import numpy as np
from config import config
from evaluate import get_evaluations_final
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
    
    
def get_optimizer(network, config):        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                                      lr=config['lr'],
                                                      weight_decay=config['l2_reg'])
    return optimizer


class ID_Bank(object):
    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        
        self.user_index_id = {}
        self.item_index_id = {}
        
        # 0 is for padding. Encode since 1. 
        self.last_item_index = 1
        self.last_user_index = 0
        
    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.user_index_id[self.last_user_index] = user_id
            self.last_user_index += 1
        return self.user_id_index[user_id]
    
    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.item_index_id[self.last_item_index] = item_id
            self.last_item_index += 1
        return self.item_id_index[item_id]
    
    def query_user_id(self, user_index):
        if user_index in self.user_index_id:
            return user_index_id[user_index]
        else:
            print(f'USER index {user_index} is not valid')
            return 'erro'
        
    def query_item_id(self, item_index):
        if item_index in self.item_index_id:
            return self.item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid')
            return 'erro'


def eval_model_ACMR(model, df, config, dtype, loader_generator, userid='userId', itemid='itemId'):
    # test_dataloader must shuffle=False
    model.eval()
    mkt_group = df.groupby('market')
    for mkt, value in mkt_group:
        test_df = value
        task_rec_all = []
        task_unq_users = set()
        probs = []
        test_dataloader = loader_generator.get_loader(dtype)
        for data in test_dataloader:
            with torch.no_grad():
                x, mkt, his_num, his_mask, target = data
                x, mkt, his_num, his_mask, target= x.to(config['device']), mkt.to(config['device']), his_num.to(config['device'])\
                                        ,his_mask.to(config['device']), target.to(config['device'])
                batch_scores = model(x, mkt, his_num, his_mask)
                batch_scores = batch_scores.squeeze().detach().cpu().numpy()
            probs.extend(list(batch_scores))
        test_pred = {}
        test_true = {}
        test_df['predict'] = probs
        test_group = test_df.groupby(userid)
        for u, v in test_group:
            tmp_pred = {}
            tmp_true = {}
            ratings = v['rate'].to_list()
            its = v[itemid].to_list()
            preds_t = v['predict'].to_list()
            for i in range(len(ratings)):
                tmp_true[str(its[i])] = int(ratings[i])
                tmp_pred[str(its[i])] = preds_t[i]
            test_pred[str(u)] = tmp_pred
            test_true[str(u)] = tmp_true 
    task_ov = get_evaluations_final(test_pred, test_true, dtype)
    return task_ov
               
               
class DataGenerator(object):
    '''
    single=True: data from src markets are all added into train set
    single=False: valid and test data comes from every mkt
    The src and tgt markets need to be mutually exclusive. The market data in tgt will be separated into the training, testing and validation set by leave-one-out method, and all the data in src will be put into the training set
    '''
    def __init__(self, src_markets, target_market, id_bank, shuffle=True, neg_num=4):
        self.id_bank = id_bank
        self.src_mkt = src_markets
        self.tgt_mkt = target_market
        self.all_mkt = src_markets+target_market
        self.mkt_num = len(set(src_markets+target_market))
        self.mkt_dict = self.get_mkt_dict()
        self.shuffle = shuffle
        self.neg_num = neg_num
        self.single = False
        if len(target_market) <= 1:
            self.single = True
        #get data
        self.data = self.load_data()
        self.item_pool = set(self.id_bank.item_index_id.keys())
        
    def get_mkt_dict(self):
        mkt_dict = {}
        for mkt in self.all_mkt:
            mkt_dict[mkt] = len(mkt_dict)
        return mkt_dict
    
    def load_data(self):
        data = {}
        for mkt in self.all_mkt:
            mkt_data = pd.read_csv(f'./data/{mkt}_5core.txt', sep=' ', usecols=['userId', 'itemId', 'rate'])
            mkt_data['market'] = self.mkt_dict[mkt]
            # transform id to idx
            mkt_data['userId'] = mkt_data['userId'].apply(lambda x: self.id_bank.query_user_index(x))
            mkt_data['itemId'] = mkt_data['itemId'].apply(lambda x: self.id_bank.query_item_index(x))
            # norm ratings
            mkt_data['rate'] = [self.normalize(cvote) for cvote in mkt_data['rate'].values.tolist()]
            self.statistic(mkt_data, mkt)
            pos_data = mkt_data[mkt_data['rate'] == 1.0]
            if self.shuffle:
                pos_data = pos_data.sample(frac = 1.0).reset_index(drop=True)
            data[mkt] = pos_data
        return data
            
    def normalize(self, score):
        if score >= 1.0:
            return 1.0
        else:
            return 0.0
        
    def statistic(self, df, mkt):
        user_num = len(df['userId'].unique())
        item_num = len(df['itemId'].unique())
        sparse = len(df)/(user_num*item_num)
        pos = df[df['rate'] == 1.0]
        interaction = len(df)
        print(f'{mkt}: users num={user_num}, item num={item_num}, ratings={interaction}, sparsity={round(sparse, 4)}')
        return
    
    def split(self, df):
        by_userid_group = df.groupby("userId")
        splits = ['remove'] * len(df)
        for usrid, indice in by_userid_group.groups.items():
            cur_item_list = list(indice)
            train_up_indx = len(cur_item_list)-2
            valid_up_index = len(cur_item_list)-1
            for iind in cur_item_list[:train_up_indx]:
                splits[iind] = 'train'
            for iind in cur_item_list[train_up_indx:valid_up_index]:
                splits[iind] = 'valid'
            for iind in cur_item_list[valid_up_index:]:
                splits[iind] = 'test'
        df['split'] = splits
        df = df[df['split']!='remove']
        df.reset_index(drop=True, inplace=True)
        train = df[df['split']=='train']
        valid = df[df['split']=='valid']
        test = df[df['split']=='test']
        return train.drop('split', 1), valid.drop('split', 1), test.drop('split', 1)
    
    def neg_sample(self, df, neg_num, dtype='train'):
        by_userid_group = df.groupby("userId")
        negs = []
        for userid, group_frame in by_userid_group:
            mkt = group_frame['market'].values.tolist()[0]
#             pos_itemids = set(group_frame['itemId'].values.tolist())
            pos_itemids = self.rated_items[userid] | set(group_frame['itemId'].values.tolist())
            neg_itemids = self.item_pool - pos_itemids
            if dtype == 'train':
                neg_itemids_sample = random.sample(neg_itemids, min(len(neg_itemids), len(pos_itemids)*neg_num))   
            else:
                neg_itemids_sample = random.sample(neg_itemids, neg_num)
            for n in neg_itemids_sample:
                row = [userid, n, 0.0, mkt]
                negs.append(row)
        negs_df = pd.DataFrame(negs, columns=['userId', 'itemId', 'rate', 'market'])
        df_pos_neg = pd.concat((df, negs_df), 0).sample(frac = 1.0).reset_index(drop=True)
        return df_pos_neg
    
    def generate_data(self):
        '''
        generate train, valid, test
        valid, test: 1 pos sample + 99 neg sample
        '''
        train = pd.DataFrame()
#         valid = pd.DataFrame()
#         test = pd.DataFrame()
        for mkt in self.src_mkt:
            train = pd.concat((train, self.data[mkt]), 0)
        for mkt in self.tgt_mkt:
            train_tgt, valid_tgt, test_tgt = self.split(self.data[mkt])
            train = pd.concat((train, train_tgt), 0)
            valid = valid_tgt
            test = test_tgt
            
        self.rated_items = {}
        train_group = train.groupby('userId')
        for u, v in train_group:
            self.rated_items[int(u)] = set(v['itemId'].to_list())

        train = self.neg_sample(train, self.neg_num)
        valid = self.neg_sample(valid, 99, 'valid')
        test = self.neg_sample(test, 99, 'test')
        return train, valid, test
    
class ACMR_loader(object):
    def __init__(self, train, valid, test, config):
        self.train = train
        self.valid = valid
        self.test = test
        self.config = config
        self.user_hist = self.get_history()
        
    def get_history(self):
        pos = self.train[self.train['rate']==1.0]
        pos_group = pos.groupby('userId')
        user_hist = {}
        avg_len = []
        for u, v in pos_group:
            his = list(v['itemId'])
            user_hist[u] = his
            avg_len.append(len(his))
        print('mean seq length:', round(np.mean(avg_len)))
        print('min seq length:', min(avg_len))
        print('max seq length:', max(avg_len))
        return user_hist
    
    def get_loader(self, dtype):
        if dtype == 'train':
            df = self.train
        elif dtype == 'valid':
            df = self.valid
        else:
            df = self.test
        x, mkt, his_num, his_mask, target = [], [], [], [], []
        for i, row in df.iterrows():
            his_seq = self.user_hist[row['userId']]
            pad_len = 0

            if len(his_seq)<=(config['bert_max_len']-1):
                pad_len = config['bert_max_len'] - 1 - len(his_seq)
                his = his_seq + [0]*pad_len
            else:
                his = random.sample(his_seq , config['bert_max_len']-1)
                
            his_num.append(min(len(his_seq) ,config['bert_max_len']-1))
            x.append(his+[int(row['itemId'])]) 
            mkt.append([row['market']]*config['bert_max_len'])
            target.append(int(row['rate']))
            his_mask.append([1]*(config['bert_max_len']-1-pad_len)+[0]*pad_len)
        dataset = TensorDataset(torch.LongTensor(x), torch.LongTensor(mkt), torch.FloatTensor(his_num), torch.FloatTensor(his_mask), torch.FloatTensor(target))
        if dtype == 'train':
            return DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=False)
    
# Checkpoints
def save(model, config):
    torch.save(model.state_dict(),config['save_path'])
    save = config['save_path']
    print(f'best model save at {save}')
def load_model(model, config):
    path = config['save_path']
    print(f'load model from: {path}')
    model.load_state_dict(torch.load(path))
    return model