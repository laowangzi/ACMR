config = {}
# 'ca', 'fr', 'in', 'jp', 'mx', 'uk', 'de'
all_mkts = set(['in', 'jp', 'mx', 'fr', 'uk', 'de', 'ca'])
tgt_mkt = ['uk']
src_mkt = list(all_mkts - set(tgt_mkt))

config['tgt_mkt'] = tgt_mkt
config['src_mkt'] = src_mkt
config['neg_num'] = 4


config['patience'] = 3
config['latent_dim'] = 32
config['dropout'] = 0.3
config['lr'] = 1e-3
config['l2_reg'] = 1e-7
config['num_epoch'] = 50

config['device'] = 'cuda:0'
config['batch_size'] = 1024
config['layers'] = [config['latent_dim']*2, config['latent_dim']*4, config['latent_dim']*2, config['latent_dim']]
config['latent_dim_mf'] = config['latent_dim']
#bert
config['bert_max_len'] = 51
config['bert_num_blocks'] = 4
config['bert_num_heads'] = 8
config['num_mkts'] = len(set(tgt_mkt+src_mkt))



    
    
 