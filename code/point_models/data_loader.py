import pickle as pkl
import time
import numpy as np


DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 50

DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 50

DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 50

class DataLoaderUserSeq(object):
    def __init__(self, batch_size, max_len, target_file, user_seq_file, neg_sample_num, user_feat_dict_file, item_feat_dict_file):
        self.batch_size = batch_size
        self.max_len = max_len
        self.neg_sample_num = neg_sample_num
        if self.batch_size % (1 + self.neg_sample_num) != 0:
            print('batch size should be time of {}'.format(1 + self.neg_sample_num))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / (1 + self.neg_sample_num))

        self.target_f = open(target_file)
        self.user_seq_f = open(user_seq_file)

        self.user_feat_dict = None
        self.item_feat_dict = None

        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        if item_feat_dict_file != None:
            with open(item_feat_dict_file, 'rb') as f:
                self.item_feat_dict = pkl.load(f)

    def __iter__(self):
        return self
    
    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        label_batch = []
        user_seq_batch = []
        user_seq_len_batch = []

        for i in range(self.batch_size2line_num):
            target_line = self.target_f.readline()
            if target_line == '':
                raise StopIteration
            user_seq_line = self.user_seq_f.readline()
            target_line_split_list = target_line[:-1].split(',')
            uid, iids = target_line_split_list[0], target_line_split_list[1:(2 + self.neg_sample_num)]
            
            user_seq_list = [iid for iid in user_seq_line[:-1].split(',')]
            user_seq_one = []
            
            for iid in user_seq_list:
                user_seq_one.append(int(iid))
            if len(user_seq_one) < self.max_len:
                user_seq_one += [user_seq_one[-1]] * (self.max_len - len(user_seq_one))
            else:
                user_seq_one = user_seq_one[-self.max_len:]

            # add feat
            if self.item_feat_dict == None:
                user_seq_one = [[iid] for iid in user_seq_one]
            else:
                user_seq_one = [[iid] + self.item_feat_dict[str(iid)] for iid in user_seq_one]
            for j in range(len(iids)):
                if j == 0:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
                if self.user_feat_dict == None:
                    target_user_batch.append([int(uid)])
                else:
                    target_user_batch.append([int(uid)] + self.user_feat_dict[uid])
                if self.item_feat_dict == None:
                    target_item_batch.append([int(iids[j])])
                else:
                    target_item_batch.append([int(iids[j])] + self.item_feat_dict[iids[j]])
                user_seq_len_batch.append(len(user_seq_list))
                user_seq_batch.append(user_seq_one)
                
        return user_seq_batch, user_seq_len_batch, target_user_batch, target_item_batch, label_batch

class DataLoaderDualSeq(object):
    def __init__(self, batch_size, max_len, target_file, user_seq_file, item_seq_file, neg_sample_num, user_feat_dict_file, item_feat_dict_file):
        self.batch_size = batch_size
        self.max_len = max_len
        self.neg_sample_num = neg_sample_num
        if self.batch_size % (1 + self.neg_sample_num) != 0:
            print('batch size should be time of {}'.format(1 + self.neg_sample_num))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / (1 + self.neg_sample_num))

        self.target_f = open(target_file)
        self.user_seq_f = open(user_seq_file)
        self.item_seq_f = open(item_seq_file)
        self.user_feat_dict = None
        self.item_feat_dict = None

        if user_feat_dict_file != None:
            with open(user_feat_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        if item_feat_dict_file != None:
            with open(item_feat_dict_file, 'rb') as f:
                self.item_feat_dict = pkl.load(f)

    def __iter__(self):
        return self
    
    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        label_batch = []
        user_seq_batch = []
        user_seq_len_batch = []
        item_seq_batch = []
        item_seq_len_batch = []

        for i in range(self.batch_size2line_num):
            target_line = self.target_f.readline()
            if target_line == '':
                raise StopIteration
            user_seq_line = self.user_seq_f.readline()
            item_seqs_line = self.item_seq_f.readline()

            target_line_split_list = target_line[:-1].split(',')
            uid, iids = target_line_split_list[0], target_line_split_list[1:(2 + self.neg_sample_num)]
            
            user_seq_list = [iid for iid in user_seq_line[:-1].split(',')]
            user_seq_one = []
            item_seqs_list = [seq.split(',') for seq in item_seqs_line[:-1].split('\t')]
            item_seqs_one = []

            for iid in user_seq_list:
                user_seq_one.append(int(iid))
            if len(user_seq_one) < self.max_len:
                user_seq_one += [user_seq_one[-1]] * (self.max_len - len(user_seq_one))
            else:
                user_seq_one = user_seq_one[-self.max_len:]
            
            # add feat
            if self.item_feat_dict == None:
                user_seq_one = [[iid] for iid in user_seq_one]
            else:
                user_seq_one = [[iid] + self.item_feat_dict[str(iid)] for iid in user_seq_one]

            for seq in item_seqs_list:
                item_seq_one = []
                for uid in seq:
                    item_seq_one.append(int(uid))
                if len(item_seq_one) < self.max_len:
                    item_seq_one += [item_seq_one[-1]] * (self.max_len - len(item_seq_one))
                else:
                    item_seq_one = item_seq_one[-self.max_len:]
                # add feat
                if self.user_feat_dict == None:
                    item_seq_one = [[uid] for uid in item_seq_one]
                else:
                    item_seq_one = [[uid] + self.user_feat_dict[str(uid)] for uid in item_seq_one]
                item_seqs_one.append(item_seq_one)

            for j in range(len(iids)):
                if j == 0:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
                if self.user_feat_dict == None:
                    target_user_batch.append([int(uid)])
                else:
                    target_user_batch.append([int(uid)] + self.user_feat_dict[uid])
                if self.item_feat_dict == None:
                    target_item_batch.append([int(iids[j])])
                else:
                    target_item_batch.append([int(iids[j])] + self.item_feat_dict[iids[j]])
                user_seq_len_batch.append(len(user_seq_list))
                user_seq_batch.append(user_seq_one)
                item_seq_len_batch.append(len(item_seqs_list[j]))
                item_seq_batch.append(item_seqs_one[j])
                
        return user_seq_batch, user_seq_len_batch, item_seq_batch, item_seq_len_batch, target_user_batch, target_item_batch, label_batch

if __name__ == "__main__":
    data_loader = DataLoaderDualSeq(100, 
                                    300, 
                                    DATA_DIR_CCMR + 'target_38.txt',
                                    DATA_DIR_CCMR + 'train_user_hist_seq_38.txt',
                                    DATA_DIR_CCMR + 'train_item_hist_seq_38.txt', 
                                    99,                                
                                    None,
                                    DATA_DIR_CCMR + 'remap_movie_info_dict.pkl')
    
    t = time.time()
    for batch_data in data_loader:
        print(np.array(batch_data[0]).shape)
        print(np.array(batch_data[1]).shape)
        print(np.array(batch_data[2]).shape)
        print(np.array(batch_data[3]).shape)
        print(np.array(batch_data[4]).shape)
        print(np.array(batch_data[5]).shape)
        print(np.array(batch_data[6]).shape)
        
        print('time of batch: {}'.format(time.time()-t))
        t = time.time()
        