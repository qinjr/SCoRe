import random
import pickle as pkl
import time
import numpy as np
import datetime
import sys

NEG_SAMPLE_NUM = 9

# CCMR dataset parameters
DATA_DIR_CCMR = '../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 300

# Taobao dataset parameters
DATA_DIR_Taobao = '../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 300

# Tmall dataset parameters
DATA_DIR_Tmall = '../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 300

def gen_user_hist_seq_file(in_file, out_file, user_hist_dict_file, max_len):
    with open(user_hist_dict_file, 'rb') as f:
        user_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            if uid in user_hist_dict:
                user_hist_list = user_hist_dict[uid]
                if len(user_hist_list) > max_len:
                    user_hist_list = user_hist_list[-max_len:]
                else:
                    user_hist_list = user_hist_list
            else:
                exit(1)
            newlines.append(','.join(user_hist_list) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('gen {} completed'.format(out_file))

def gen_item_hist_seq_file(in_file, out_file, item_hist_dict_file, max_len):
    with open(item_hist_dict_file, 'rb') as f:
        item_hist_dict = pkl.load(f)
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            iids = line[:-1].split(',')[1:]
            item_seq_list = []
            for iid in iids:
                if iid in item_hist_dict:
                    item_hist_list = item_hist_dict[iid]
                    # if mode == 'test':
                    if len(item_hist_list) > max_len:
                        item_hist_list = item_hist_list[-max_len:]
                    else:
                        item_hist_list = item_hist_list
                else:
                    item_hist_list = ['0']
                item_seq_list.append(','.join(item_hist_list))
            newlines.append('\t'.join(item_seq_list) + '\n')
    
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('gen {} completed'.format(out_file))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_38.txt', DATA_DIR_CCMR + 'train_user_hist_seq_38.txt', DATA_DIR_CCMR + 'user_hist_dict_38.pkl', MAX_LEN_CCMR)
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_39.txt', DATA_DIR_CCMR + 'validation_user_hist_seq_39.txt', DATA_DIR_CCMR + 'user_hist_dict_39.pkl', MAX_LEN_CCMR)
        gen_user_hist_seq_file(DATA_DIR_CCMR + 'target_40.txt', DATA_DIR_CCMR + 'test_user_hist_seq_40.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', MAX_LEN_CCMR)

        gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_38.txt', DATA_DIR_CCMR + 'train_item_hist_seq_38.txt', DATA_DIR_CCMR + 'item_hist_dict_38.pkl', MAX_LEN_CCMR)
        gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_39.txt', DATA_DIR_CCMR + 'validation_item_hist_seq_39.txt', DATA_DIR_CCMR + 'item_hist_dict_39.pkl', MAX_LEN_CCMR)
        gen_item_hist_seq_file(DATA_DIR_CCMR + 'target_40.txt', DATA_DIR_CCMR + 'test_item_hist_seq_40.txt', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', MAX_LEN_CCMR)
    elif dataset == 'taobao':
        # Taobao
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_6.txt', DATA_DIR_Taobao + 'train_user_hist_seq_6.txt', DATA_DIR_Taobao + 'user_hist_dict_6.pkl', MAX_LEN_Taobao)
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_7.txt', DATA_DIR_Taobao + 'validation_user_hist_seq_7.txt', DATA_DIR_Taobao + 'user_hist_dict_7.pkl', MAX_LEN_Taobao)
        gen_user_hist_seq_file(DATA_DIR_Taobao + 'target_8.txt', DATA_DIR_Taobao + 'test_user_hist_seq_8.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl', MAX_LEN_Taobao)

        gen_item_hist_seq_file(DATA_DIR_Taobao + 'target_6.txt', DATA_DIR_Taobao + 'train_item_hist_seq_6.txt', DATA_DIR_Taobao + 'item_hist_dict_6.pkl', MAX_LEN_Taobao)
        gen_item_hist_seq_file(DATA_DIR_Taobao + 'target_7.txt', DATA_DIR_Taobao + 'validation_item_hist_seq_7.txt', DATA_DIR_Taobao + 'item_hist_dict_7.pkl', MAX_LEN_Taobao)
        gen_item_hist_seq_file(DATA_DIR_Taobao + 'target_8.txt', DATA_DIR_Taobao + 'test_item_hist_seq_8.txt', DATA_DIR_Taobao + 'item_hist_dict_8.pkl', MAX_LEN_Taobao)
    elif dataset == 'tmall':
        # Tmall
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_9.txt', DATA_DIR_Tmall + 'train_user_hist_seq_9.txt', DATA_DIR_Tmall + 'user_hist_dict_9.pkl', MAX_LEN_Tmall)
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_10.txt', DATA_DIR_Tmall + 'validation_user_hist_seq_10.txt', DATA_DIR_Tmall + 'user_hist_dict_10.pkl', MAX_LEN_Tmall)
        gen_user_hist_seq_file(DATA_DIR_Tmall + 'target_11.txt', DATA_DIR_Tmall + 'test_user_hist_seq_11.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', MAX_LEN_Tmall)
        
        gen_item_hist_seq_file(DATA_DIR_Tmall + 'target_9.txt', DATA_DIR_Tmall + 'train_item_hist_seq_9.txt', DATA_DIR_Tmall + 'item_hist_dict_9.pkl', MAX_LEN_Tmall)
        gen_item_hist_seq_file(DATA_DIR_Tmall + 'target_10.txt', DATA_DIR_Tmall + 'validation_item_hist_seq_10.txt', DATA_DIR_Tmall + 'item_hist_dict_10.pkl', MAX_LEN_Tmall)
        gen_item_hist_seq_file(DATA_DIR_Tmall + 'target_11.txt', DATA_DIR_Tmall + 'test_item_hist_seq_11.txt', DATA_DIR_Tmall + 'item_hist_dict_11.pkl', MAX_LEN_Tmall)

    else:
        print('WRONG DATASET: {}'.format(dataset))