import sys
import math
import pandas as pd
import numpy as np
import random
from config import config
from tqdm import tqdm
import pickle as pkl
import json
import os
import random


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    #true:2^1-1=1, false:2^0-1=0
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    idcg = dcg_score(y_true, y_true, k)
    dcg = dcg_score(y_true, y_score, k)
    return dcg/idcg

def recall(y_truth, y_pred, k):
    order = np.argsort(y_pred)[::-1]
    sort_list = np.take(y_truth, order[:k])
    #our experiment is the leave-one-out setting
    if sum(sort_list) >= 1:
        return 1
    return 0



def get_evaluations_final(y_pred, y_true, dtype):
    recall_5 = []
    recall_10 = []
    ndcg_5 = []
    ndcg_10 = []
    map_5 = []
    map_10 = []    
    for k in y_pred.keys():
        recall_5.append(recall(np.array(list(y_true[k].values())), np.array(list(y_pred[k].values())), 5))
        recall_10.append(recall(np.array(list(y_true[k].values())), np.array(list(y_pred[k].values())), 10))
        ndcg_5.append(ndcg_score(np.array(list(y_true[k].values())), np.array(list(y_pred[k].values())), 5))
        ndcg_10.append(ndcg_score(np.array(list(y_true[k].values())), np.array(list(y_pred[k].values())), 10))
    result = {}
    result['recall@5'] = np.mean(recall_5)
    result['recall@10'] = np.mean(recall_10)
    result['ndcg@5'] = np.mean(ndcg_5)
    result['ndcg@10'] = np.mean(ndcg_10)
    print(dtype, end=': ')
    for k, v in result.items():
        print(f'{k}: {round(v, 4)}', end=', ')
    print('')
    return result