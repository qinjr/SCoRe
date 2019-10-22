import random
import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing
import datetime
import sys

random.seed(11)

NEG_SAMPLE_NUM = 99
SECONDS_PER_DAY = 24*3600
# CCMR dataset parameters
DATA_DIR_CCMR = '../score-data/CCMR/feateng/'
OBJ_PER_TIME_SLICE_CCMR = 10
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129
USER_PER_COLLECTION_CCMR = 1000
ITEM_PER_COLLECTION_CCMR = 100
START_TIME_CCMR = 1116432000
START_TIME_IDX_CCMR = 0
TIME_DELTA_CCMR = 90

# Taobao dataset parameters
DATA_DIR_Taobao = '../score-data/Taobao/feateng/'
OBJ_PER_TIME_SLICE_Taobao = 10
USER_NUM_Taobao = 984080
ITEM_NUM_Taobao = 4049268
USER_PER_COLLECTION_Taobao = 500
ITEM_PER_COLLECTION_Taobao = 500
START_TIME_Taobao = int(time.mktime(datetime.datetime.strptime('2017-11-25', "%Y-%m-%d").timetuple()))
START_TIME_IDX_Taobao = 0
TIME_DELTA_Taobao = 1

# Tmall dataset parameters
DATA_DIR_Tmall = '../score-data/Tmall/feateng/'
OBJ_PER_TIME_SLICE_Tmall = 10
USER_NUM_Tmall = 424170
ITEM_NUM_Tmall = 1090390
USER_PER_COLLECTION_Tmall = 200
ITEM_PER_COLLECTION_Tmall = 500
START_TIME_Tmall = int(time.mktime(datetime.datetime.strptime('2015-5-1', "%Y-%m-%d").timetuple()))
START_TIME_IDX_Tmall = 0
TIME_DELTA_Tmall = 15

class TargetGen(object):
    def __init__(self, user_neg_dict_file, db_name, user_num, item_num, user_per_collection,
                item_per_collection, start_time, start_time_idx, time_delta):
        if user_neg_dict_file != None:
            with open(user_neg_dict_file, 'rb') as f:
                self.user_neg_dict = pkl.load(f)  
        else:
            self.user_neg_dict = {}

        url = "mongodb://localhost:27017/"
        client = pymongo.MongoClient(url)
        db = client[db_name]
        self.user_num = user_num
        self.item_num = item_num
        
        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection

        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1

        self.user_colls = [db['user_%d'%(i)] for i in range(user_coll_num)]
        self.item_colls = [db['item_%d'%(i)] for i in range(item_coll_num)]
        
        self.start_time = start_time
        self.start_time_idx = start_time_idx
        self.time_delta = time_delta

    def gen_user_neg_items(self, uid, neg_sample_num, start_iid, end_iid, pop_items):
        if str(uid) in self.user_neg_dict:
            user_neg_list = self.user_neg_dict[str(uid)]
        else:
            user_neg_list = []
        
        if len(user_neg_list) >= neg_sample_num:
            return user_neg_list[:neg_sample_num]
        else:
            if pop_items == None:
                for i in range(neg_sample_num - len(user_neg_list)):
                    user_neg_list.append(str(random.randint(start_iid, end_iid)))
                return user_neg_list
            else:
                pop_items_len = len(pop_items)
                for i in range(neg_sample_num - len(user_neg_list)):
                    user_neg_list.append(pop_items[random.randint(0, pop_items_len-1)])
                return user_neg_list

    def gen_target_file(self, neg_sample_num, target_file, user_hist_dict_file, pred_time, pop_items_file = None):
        if pop_items_file != None:
            with open(pop_items_file, 'rb') as f:
                pop_items = pkl.load(f)
        else:
            pop_items = None
        
        with open(user_hist_dict_file, 'rb') as f:
            user_hist_dict = pkl.load(f)

        target_lines = []
        for user_coll in self.user_colls:
            cursor = user_coll.find({})
            for user_doc in cursor:
                if user_doc['1hop'][pred_time] != []:
                    uid = user_doc['uid']
                    if str(uid) in user_hist_dict:
                        pos_iids = user_doc['1hop'][pred_time]
                        pos_iid = pos_iids[0]
                        neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num, pop_items)
                        target_lines.append(','.join([str(uid), str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))

    def gen_user_item_hist_dict_ccmr(self, hist_file, user_hist_dict_file, item_hist_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}

        # load and construct dicts
        with open(hist_file, 'r') as f:
            for line in f:
                uid, iid, _, time_str = line[:-1].split(',')
                uid = str(int(uid) + 1)
                iid = str(int(iid) + 1 + self.user_num)
                time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if time_idx < self.start_time_idx:
                    continue
                if time_idx >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, time_int)]
                else:
                    user_hist_dict[uid].append((iid, time_int))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, time_int)]
                else:
                    item_hist_dict[iid].append((uid, time_int))
            print('dicts construct completed')
        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)

    def gen_user_item_hist_dict_taobao(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            for line in f:
                uid, iid, _, timestamp_str = line[:-1].split(',')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                timestamp = int(timestamp_str)
                time_idx = int((timestamp - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, timestamp)]
                else:
                    user_hist_dict[uid].append((iid, timestamp))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, timestamp)]
                else:
                    item_hist_dict[iid].append((uid, timestamp))
            print('dicts construct completed')

        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)

    def gen_user_item_hist_dict_tmall(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                uid, iid, cid, sid, bid, date, btypeid, aid, gid = line[:-1].split(',')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                time_int = int(time.mktime(datetime.datetime.strptime('2015'+date, "%Y%m%d").timetuple()))
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, time_int)]
                else:
                    user_hist_dict[uid].append((iid, time_int))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, time_int)]
                else:
                    item_hist_dict[iid].append((uid, time_int))
            print('dicts construct completed')

        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)
    
    def gen_pop_items(self, pop_items_file, pop_standard, max_iid):
        pop_items = []
        for item_coll in self.item_colls:
            cursor = item_coll.find({})
            for item_doc in cursor:
                num_not_null_slice = 0
                for nei in item_doc['1hop']:
                    if nei != []:
                        num_not_null_slice += 1
                if num_not_null_slice >= pop_standard and item_doc['iid'] <= max_iid:
                    pop_items.append(str(item_doc['iid']))
        print('num of pop_items: {}'.format(len(pop_items)))
        with open(pop_items_file, 'wb') as f:
            pkl.dump(pop_items, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        tg = TargetGen(DATA_DIR_CCMR + 'user_neg_dict.pkl', 'ccmr_1hop', user_num = USER_NUM_CCMR,
                    item_num = ITEM_NUM_CCMR, user_per_collection = USER_PER_COLLECTION_CCMR,
                    item_per_collection = ITEM_PER_COLLECTION_CCMR, start_time = START_TIME_CCMR, 
                    start_time_idx = START_TIME_IDX_CCMR, time_delta = TIME_DELTA_CCMR)
        # gen hist dict file: train, validation, test
        tg.gen_user_item_hist_dict_ccmr(DATA_DIR_CCMR + 'rating_pos.csv', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', DATA_DIR_CCMR + 'item_hist_dict_40.pkl', 40)
        tg.gen_user_item_hist_dict_ccmr(DATA_DIR_CCMR + 'rating_pos.csv', DATA_DIR_CCMR + 'user_hist_dict_39.pkl', DATA_DIR_CCMR + 'item_hist_dict_39.pkl', 39)
        tg.gen_user_item_hist_dict_ccmr(DATA_DIR_CCMR + 'rating_pos.csv', DATA_DIR_CCMR + 'user_hist_dict_38.pkl', DATA_DIR_CCMR + 'item_hist_dict_38.pkl', 38)

        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_40.txt', DATA_DIR_CCMR + 'user_hist_dict_40.pkl', 40, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_39.txt', DATA_DIR_CCMR + 'user_hist_dict_39.pkl', 39, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_CCMR + 'target_38.txt', DATA_DIR_CCMR + 'user_hist_dict_38.pkl', 38, None)
    
    elif dataset == 'taobao':
        # Taobao
        tg = TargetGen(None, 'taobao_1hop', user_num = USER_NUM_Taobao,
                    item_num = ITEM_NUM_Taobao, user_per_collection = USER_PER_COLLECTION_Taobao,
                    item_per_collection = ITEM_PER_COLLECTION_Taobao, start_time = START_TIME_Taobao, 
                    start_time_idx = START_TIME_IDX_Taobao, time_delta = TIME_DELTA_Taobao)
        tg.gen_pop_items(DATA_DIR_Taobao + 'pop_items.pkl', 6, 1 + USER_NUM_Taobao + ITEM_NUM_Taobao)

        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Taobao + 'filtered_user_behavior.txt', DATA_DIR_Taobao + 'user_hist_dict_6.pkl', DATA_DIR_Taobao + 'item_hist_dict_6.pkl', DATA_DIR_Taobao + 'remap_dict.pkl', 6)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Taobao + 'filtered_user_behavior.txt', DATA_DIR_Taobao + 'user_hist_dict_7.pkl', DATA_DIR_Taobao + 'item_hist_dict_7.pkl', DATA_DIR_Taobao + 'remap_dict.pkl', 7)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Taobao + 'filtered_user_behavior.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl', DATA_DIR_Taobao + 'item_hist_dict_8.pkl', DATA_DIR_Taobao + 'remap_dict.pkl', 8)
        
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Taobao + 'target_6.txt', DATA_DIR_Taobao + 'user_hist_dict_6.pkl', 6, DATA_DIR_Taobao + 'pop_items.pkl')
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Taobao + 'target_7.txt', DATA_DIR_Taobao + 'user_hist_dict_7.pkl', 7, DATA_DIR_Taobao + 'pop_items.pkl')
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Taobao + 'target_8.txt', DATA_DIR_Taobao + 'user_hist_dict_8.pkl', 8, DATA_DIR_Taobao + 'pop_items.pkl')
        
        
    elif dataset == 'tmall':
        # Tmall
        tg = TargetGen(None, 'tmall_1hop', user_num = USER_NUM_Tmall,
                    item_num = ITEM_NUM_Tmall, user_per_collection = USER_PER_COLLECTION_Tmall,
                    item_per_collection = ITEM_PER_COLLECTION_Tmall, start_time = START_TIME_Tmall, 
                    start_time_idx = START_TIME_IDX_Tmall, time_delta = TIME_DELTA_Tmall)
        tg.gen_pop_items(DATA_DIR_Tmall + 'pop_items.pkl', 9, 1 + USER_NUM_Tmall + ITEM_NUM_Tmall)

        tg.gen_user_item_hist_dict_tmall(DATA_DIR_Tmall + 'joined_user_behavior.csv', DATA_DIR_Tmall + 'user_hist_dict_9.pkl', DATA_DIR_Tmall + 'item_hist_dict_9.pkl', DATA_DIR_Tmall + 'remap_dict.pkl', 9)
        tg.gen_user_item_hist_dict_tmall(DATA_DIR_Tmall + 'joined_user_behavior.csv', DATA_DIR_Tmall + 'user_hist_dict_10.pkl', DATA_DIR_Tmall + 'item_hist_dict_10.pkl', DATA_DIR_Tmall + 'remap_dict.pkl', 10)
        tg.gen_user_item_hist_dict_tmall(DATA_DIR_Tmall + 'joined_user_behavior.csv', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', DATA_DIR_Tmall + 'item_hist_dict_11.pkl', DATA_DIR_Tmall + 'remap_dict.pkl', 11)

        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Tmall + 'target_9.txt', DATA_DIR_Tmall + 'user_hist_dict_9.pkl', 9, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Tmall + 'target_10.txt', DATA_DIR_Tmall + 'user_hist_dict_10.pkl', 10, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Tmall + 'target_11.txt', DATA_DIR_Tmall + 'user_hist_dict_11.pkl', 11, None)
        
    else:
        print('WRONG DATASET: {}'.format(dataset))

