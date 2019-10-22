import pymongo
import pickle as pkl
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

random.seed(11)

SECONDS_PER_DAY = 24 * 3600

# CCMR parameters
DATA_DIR_CCMR = '../score-data/CCMR/feateng/'
USER_PER_COLLECTION_CCMR = 1000
ITEM_PER_COLLECTION_CCMR = 100
START_TIME_CCMR = 0
MAX_1HOP_CCMR = 10
MAX_2HOP_CCMR = 100
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129
TIME_SLICE_NUM_CCMR = 41

# Taobao parameters
DATA_DIR_Taobao = '../score-data/Taobao/feateng/'
USER_PER_COLLECTION_Taobao = 500
ITEM_PER_COLLECTION_Taobao = 500
START_TIME_Taobao = 0
MAX_1HOP_Taobao = 10
MAX_2HOP_Taobao = 100
USER_NUM_Taobao = 984080
ITEM_NUM_Taobao = 4049268
TIME_SLICE_NUM_Taobao = 9

# Tmall parameters
DATA_DIR_Tmall = '../score-data/Tmall/feateng/'
USER_PER_COLLECTION_Tmall = 200
ITEM_PER_COLLECTION_Tmall = 500
START_TIME_Tmall = 0
MAX_1HOP_Tmall = 10
MAX_2HOP_Tmall = 100
USER_NUM_Tmall = 424170
ITEM_NUM_Tmall = 1090390
TIME_SLICE_NUM_Tmall = 14


class GraphStore(object):
    def __init__(self, rating_file, user_per_collection = USER_PER_COLLECTION_CCMR, 
                item_per_collection = ITEM_PER_COLLECTION_CCMR,  start_time = START_TIME_CCMR,   
                max_1hop = MAX_1HOP_CCMR, max_2hop = MAX_2HOP_CCMR, user_num = USER_NUM_CCMR,
                item_num = ITEM_NUM_CCMR, db_1hop = 'ccmr_1hop', db_2hop = 'ccmr_2hop',
                time_slice_num = TIME_SLICE_NUM_CCMR):
        self.url = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(self.url)
        
        self.db_1hop = self.client[db_1hop]
        self.db_2hop = self.client[db_2hop]
        
        self.user_num = user_num
        self.item_num = item_num

        # input files
        self.rating_file = open(rating_file, 'r')

        # about time index
        self.time_slice_num = time_slice_num

        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection
        self.start_time = start_time
        self.max_1hop = max_1hop
        self.max_2hop = max_2hop


    def gen_user_doc(self, uid):
        user_doc = {}
        user_doc['uid'] = uid
        user_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        return user_doc

    def gen_item_doc(self, iid):
        item_doc = {}
        item_doc['iid'] = iid
        item_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        return item_doc

    def construct_coll_1hop(self):
        list_of_user_doc_list = []
        list_of_item_doc_list = []

        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1

        for i in range(user_coll_num):
            user_doc_list = []
            for uid in range(i * self.user_per_collection + 1, (i + 1) * self.user_per_collection + 1):
                user_doc_list.append(self.gen_user_doc(uid))
            list_of_user_doc_list.append(user_doc_list)

        for i in range(item_coll_num):
            item_doc_list = []
            for iid in range(i * self.item_per_collection + 1 + self.user_num, (i + 1) * self.item_per_collection + 1 + self.user_num):
                item_doc_list.append(self.gen_item_doc(iid))
            list_of_item_doc_list.append(item_doc_list)

        for line in self.rating_file:
            uid, iid, _, t_idx = line[:-1].split(',')
            list_of_user_doc_list[(int(uid) - 1) // self.user_per_collection][(int(uid) - 1) % self.user_per_collection]['1hop'][int(t_idx)].append(int(iid))
            list_of_item_doc_list[(int(iid) - self.user_num - 1) // self.item_per_collection][(int(iid) - self.user_num - 1) % self.item_per_collection]['1hop'][int(t_idx)].append(int(uid))
        
        print('user and item doc list completed')

        for i in range(len(list_of_user_doc_list)):
            self.db_1hop['user_%d'%(i)].insert_many(list_of_user_doc_list[i])
        print('user collection completed')
        for i in range(len(list_of_item_doc_list)):
            self.db_1hop['item_%d'%(i)].insert_many(list_of_item_doc_list[i])
        print('item collection completed')
    
    def construct_coll_2hop(self):
        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1
        
        user_colls = [self.db_1hop['user_%d'%i] for i in range(user_coll_num)]
        item_colls = [self.db_1hop['item_%d'%i] for i in range(item_coll_num)]

        all_user_docs = []
        all_item_docs = []
        for user_coll in user_colls:
            cursor = user_coll.find({})
            for user_doc in cursor:
                all_user_docs.append(user_doc)
        for item_coll in item_colls:
            cursor = item_coll.find({})
            for item_doc in cursor:
                all_item_docs.append(item_doc)
        print('loading 1hop graph data completed')
        

        # gen item 2hop
        print('item 2 hop gen begin')
        for i in range(item_coll_num):
            item_docs_block = []
            for iid in range(1 + self.user_num + i * self.item_per_collection, 1 + self.user_num + (i + 1) * self.item_per_collection):
                old_item_doc = all_item_docs[iid - 1 - self.user_num]
                new_item_doc = {
                    'iid': iid,
                    '1hop': old_item_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(self.start_time):
                    new_item_doc['2hop'].append([])
                    new_item_doc['degrees'].append([])
                for t in range(self.start_time, self.time_slice_num):
                    iids_2hop = []
                    degrees_2hop = []
                    
                    uids = old_item_doc['1hop'][t]
                    if len(uids) > self.max_1hop:
                        random.shuffle(uids)
                        uids = uids[:self.max_1hop]
                    for uid in uids:
                        user_doc = all_user_docs[uid - 1]
                        degree = len(user_doc['1hop'][t])
                        if degree > 1 and degree <= self.max_1hop:
                            iids_2hop += user_doc['1hop'][t]
                            degrees_2hop += [degree] * degree
                        elif degree > self.max_1hop:
                            iids_2hop += user_doc['1hop'][t][:self.max_1hop]
                            degrees_2hop += [degree] * self.max_1hop
                        else:
                            continue

                    if len(iids_2hop) > self.max_2hop:
                        idx = np.random.choice(np.arange(len(iids_2hop)), len(iids_2hop), replace=False)
                        iids_2hop = np.array(iids_2hop)[idx].tolist()[:self.max_2hop]
                        degrees_2hop = np.array(degrees_2hop)[idx].tolist()[:self.max_2hop]

                    new_item_doc['2hop'].append(iids_2hop)
                    new_item_doc['degrees'].append(degrees_2hop)

                item_docs_block.append(new_item_doc)
            self.db_2hop['item_%d'%i].insert_many(item_docs_block)
            print('item block-{} completed'.format(i))
        print('item 2 hop gen completed')

        # gen user 2hop
        print('user 2 hop gen begin')
        for i in range(user_coll_num):
            user_docs_block = []
            for uid in range(1 + i * self.user_per_collection, 1 + (i + 1) * self.user_per_collection):
                old_user_doc = all_user_docs[uid - 1]
                new_user_doc = {
                    'uid': uid,
                    '1hop': old_user_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(self.start_time):
                    new_user_doc['2hop'].append([])
                    new_user_doc['degrees'].append([])
                for t in range(self.start_time, self.time_slice_num):
                    uids_2hop = []
                    degrees_2hop = []

                    iids = old_user_doc['1hop'][t]
                    if len(iids) > self.max_1hop:
                        random.shuffle(iids)
                        iids = iids[:self.max_1hop]
                    for iid in iids:
                        item_doc = all_item_docs[iid - 1 - self.user_num]
                        degree = len(item_doc['1hop'][t])
                        if degree > 1 and degree <= self.max_1hop:
                            uids_2hop += item_doc['1hop'][t]
                            degrees_2hop += [degree] * degree
                        elif degree > self.max_1hop:
                            uids_2hop += item_doc['1hop'][t][:self.max_1hop]
                            degrees_2hop += [degree] * self.max_1hop
                        else:
                            continue
                        
                    if len(uids_2hop) > self.max_2hop:
                        idx = np.random.choice(np.arange(len(uids_2hop)), len(uids_2hop), replace=False)
                        uids_2hop = np.array(uids_2hop)[idx].tolist()[:self.max_2hop]
                        degrees_2hop = np.array(degrees_2hop)[idx].tolist()[:self.max_2hop]

                    new_user_doc['2hop'].append(uids_2hop)
                    new_user_doc['degrees'].append(degrees_2hop)


                user_docs_block.append(new_user_doc)
            self.db_2hop['user_%d'%i].insert_many(user_docs_block)
            print('user block-{} completed'.format(i))
        print('user 2 hop gen completed')


    def cal_stat(self):
        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1
        
        user_colls = [self.db_1hop['user_%d'%i] for i in range(user_coll_num)]
        item_colls = [self.db_1hop['item_%d'%i] for i in range(item_coll_num)]
        
        # calculate user doc
        hist_len_user = []
        for user_coll in user_colls:
            for user_doc in user_coll.find({}):
                for t in range(self.time_slice_num):
                    hist_len_user.append(len(user_doc['1hop'][t]))
        
        arr = np.array(hist_len_user)
        print('max user slice hist len: {}'.format(np.max(arr)))
        print('min user slice hist len: {}'.format(np.min(arr)))
        print('null slice per user: {}'.format(arr[arr == 0].size / self.user_num))
        print('small(<=5) slice per user: {}'.format(arr[arr <= 5].size / self.user_num))
        print('mean user slice(not null) hist len: {}'.format(np.mean(arr[arr > 0])))

        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)

        
        print('-------------------------------------')
        # calculate item doc
        hist_len_item = []
        for item_coll in item_colls:
            for item_doc in item_coll.find({}):
                for t in range(self.time_slice_num):
                    hist_len_item.append(len(item_doc['1hop'][t]))
        arr = np.array(hist_len_item)
        print('max item hist len: {}'.format(np.max(arr)))
        print('min item hist len: {}'.format(np.min(arr)))
        print('null per item: {}'.format(arr[arr == 0].size / self.item_num))
        print('small(<=5) per item: {}'.format(arr[arr <= 5].size / self.item_num))
        print('mean item hist(not null) len: {}'.format(np.mean(arr[arr > 0])))
        
        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # For CCMR
        gs = GraphStore(DATA_DIR_CCMR + 'remap_rating_pos_idx.csv', user_per_collection = USER_PER_COLLECTION_CCMR, 
                    item_per_collection = ITEM_PER_COLLECTION_CCMR,  start_time = START_TIME_CCMR,   
                    max_1hop = MAX_1HOP_CCMR, max_2hop = MAX_2HOP_CCMR, user_num = USER_NUM_CCMR,
                    item_num = ITEM_NUM_CCMR, db_1hop = 'ccmr_1hop', db_2hop = 'ccmr_2hop',
                    time_slice_num = TIME_SLICE_NUM_CCMR)
        gs.construct_coll_1hop()
        gs.construct_coll_2hop()
        # gs.cal_stat()
    elif dataset == 'taobao':
        # For Taobao
        gs = GraphStore(DATA_DIR_Taobao + 'remaped_user_behavior.txt', user_per_collection = USER_PER_COLLECTION_Taobao, 
                    item_per_collection = ITEM_PER_COLLECTION_Taobao,  start_time = START_TIME_Taobao,   
                    max_1hop = MAX_1HOP_Taobao, max_2hop = MAX_2HOP_Taobao, user_num = USER_NUM_Taobao,
                    item_num = ITEM_NUM_Taobao, db_1hop = 'taobao_1hop', db_2hop = 'taobao_2hop',
                    time_slice_num = TIME_SLICE_NUM_Taobao)
        gs.construct_coll_1hop()
        gs.construct_coll_2hop()
        # gs.cal_stat()
    elif dataset == 'tmall':
        # For Tmall
        gs = GraphStore(DATA_DIR_Tmall + 'remaped_user_behavior.csv', user_per_collection = USER_PER_COLLECTION_Tmall, 
                    item_per_collection = ITEM_PER_COLLECTION_Tmall,  start_time = START_TIME_Tmall,   
                    max_1hop = MAX_1HOP_Tmall, max_2hop = MAX_2HOP_Tmall, user_num = USER_NUM_Tmall,
                    item_num = ITEM_NUM_Tmall, db_1hop = 'tmall_1hop', db_2hop = 'tmall_2hop',
                    time_slice_num = TIME_SLICE_NUM_Tmall)
        gs.construct_coll_1hop()
        gs.construct_coll_2hop()
        # gs.cal_stat()
    else:
        print('WRONG DATASET: {}'.format(dataset))