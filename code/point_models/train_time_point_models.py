import os
import tensorflow as tf
import sys
from data_loader import *
from point_model import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 1000
TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 99


# for CCMR
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1)
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
MAX_LEN_CCMR = 50

# for Taobao
FEAT_SIZE_Taobao = 1 + 984080 + 4049268 + 9405
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
MAX_LEN_Taobao = 50

# for Tmall
FEAT_SIZE_Tmall = 1529672
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
MAX_LEN_Tmall = 50

def restore(data_set, target_file_test, user_seq_file_test, item_seq_file_test,
        model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, 
        lr, reg_lambda, user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum):
    print('restore begin')
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'ARNN': 
        model = ARNN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'SVD++': 
        model = SVDpp(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'SASRec': 
        model = SASRec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'DELF': 
        model = DELF(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'DEEMS': 
        model = DEEMS(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        _, _, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss = eval(model_type, model, sess, target_file_test, max_time_len, reg_lambda, user_seq_file_test, item_seq_file_test, user_feat_dict_file, item_feat_dict_file)
        # p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        # rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print('RESTORE, LOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f' % (loss, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr))
        with open('logs_{}/{}.test.result'.format(data_set, model_name), 'w') as f:
            f.write('Result Test NDCG@5: {}\n'.format(ndcg_5))
            f.write('Result Test NDCG@10: {}\n'.format(ndcg_10))
            f.write('Result Test HR@1: {}\n'.format(hr_1))
            f.write('Result Test HR@5: {}\n'.format(hr_5))
            f.write('Result Test HR@10: {}\n'.format(hr_10))
            f.write('Result Test MRR: {}\n'.format(mrr))

def get_ndcg(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_val = []
    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        ndcg_val.append(getNDCG_at_K(ranklist, pos_iids[i], 5))
    return np.mean(ndcg_val)

def getNDCG_at_K(ranklist, target_item, k):
    for i in range(k):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0

def getHR_at_K(ranklist, target_item, k):
    if target_item in ranklist[:k]:
        return 1
    else:
        return 0

def getMRR(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return 1. / (i+1)
    return 0

def get_ranking_quality(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:,0].flatten().tolist()
    ndcg_5_val = []
    ndcg_10_val = []
    hr_1_val = []
    hr_5_val = []
    hr_10_val = []
    mrr_val = []

    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        target_item = pos_iids[i]
        ndcg_5_val.append(getNDCG_at_K(ranklist, target_item, 5))
        ndcg_10_val.append(getNDCG_at_K(ranklist, target_item, 10))
        hr_1_val.append(getHR_at_K(ranklist, target_item, 1))
        hr_5_val.append(getHR_at_K(ranklist, target_item, 5))
        hr_10_val.append(getHR_at_K(ranklist, target_item, 10))
        mrr_val.append(getMRR(ranklist, target_item))
    return np.mean(ndcg_5_val), np.mean(ndcg_10_val), np.mean(hr_1_val), np.mean(hr_5_val), np.mean(hr_10_val), np.mean(mrr_val)


def eval(model_type, model, sess, target_file, max_time_len, reg_lambda, user_seq_file, item_seq_file, user_feat_dict_file, item_feat_dict_file):
    preds = []
    labels = []
    target_iids = []
    losses = []
    if model_type == 'DELF' or model_type == 'DEEMS':
        data_loader = DataLoaderDualSeq(EVAL_BATCH_SIZE, max_time_len, target_file, user_seq_file, item_seq_file, TEST_NEG_SAMPLE_NUM, user_feat_dict_file, item_feat_dict_file)
    else:    
        data_loader = DataLoaderUserSeq(EVAL_BATCH_SIZE, max_time_len, target_file, user_seq_file, TEST_NEG_SAMPLE_NUM, user_feat_dict_file, item_feat_dict_file)
    
    t = time.time()
    for batch_data in data_loader:
        pred, label, loss = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        losses.append(loss)
        if model_type == 'DELF' or model_type == 'DEEMS':
            target_iids += np.array(batch_data[5])[:,0].tolist()
        else:
            target_iids += np.array(batch_data[3])[:,0].tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return logloss, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss

def train(data_set, target_file_train, target_file_validation, user_seq_file_train, user_seq_file_validation,
        item_seq_file_train, item_seq_file_validation, model_type, train_batch_size, feature_size, 
        eb_dim, hidden_size, max_time_len, lr, reg_lambda, dataset_size, user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum):
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'ARNN': 
        model = ARNN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'SVD++': 
        model = SVDpp(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'SASRec': 
        model = SASRec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'DELF': 
        model = DELF(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    elif model_type == 'DEEMS': 
        model = DEEMS(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []
        train_losses = []
        
        vali_ndcgs_5 = []
        vali_ndcgs_10 = []
        vali_hrs_1 = []
        vali_hrs_5 = []
        vali_hrs_10 = []
        vali_mrrs = []
        vali_losses = []

        # before training process
        step = 0
        _, _, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model_type, model, sess, target_file_validation, max_time_len, reg_lambda, user_seq_file_validation, item_seq_file_validation, user_feat_dict_file, item_feat_dict_file)
        
        vali_ndcgs_5.append(vali_ndcg_5)
        vali_ndcgs_10.append(vali_ndcg_10)
        vali_hrs_1.append(vali_hr_1)
        vali_hrs_5.append(vali_hr_5)
        vali_hrs_10.append(vali_hr_10)
        vali_mrrs.append(vali_mrr)
        vali_losses.append(vali_loss)

        print("STEP %d  LOSS TRAIN: NULL  LOSS VALI: %.4f  NDCG@5 VALI: %.4f  NDCG@10 VALI: %.4f  HR@1 VALI: %.4f  HR@5 VALI: %.4f  HR@10 VALI: %.4f  MRR VALI: %.4f" % (step, vali_loss, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr))
        early_stop = False
        eval_iter_num = (dataset_size // 3) // (train_batch_size / (1 + TRAIN_NEG_SAMPLE_NUM))
        # begin training process
        for epoch in range(10):
            if early_stop:
                break
            if model_type == 'DELF' or model_type == 'DEEMS':
                data_loader = DataLoaderDualSeq(train_batch_size, max_time_len, target_file_train, user_seq_file_train, item_seq_file_train, TEST_NEG_SAMPLE_NUM, user_feat_dict_file, item_feat_dict_file)
            else:    
                data_loader = DataLoaderUserSeq(train_batch_size, max_time_len, target_file_train, user_seq_file_train, TEST_NEG_SAMPLE_NUM, user_feat_dict_file, item_feat_dict_file)
            
            for batch_data in data_loader:
                if early_stop:
                    break
                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []
                    
                    _, _, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model_type, model, sess, target_file_validation, max_time_len, reg_lambda, user_seq_file_validation, item_seq_file_validation, user_feat_dict_file, item_feat_dict_file)

                    vali_ndcgs_5.append(vali_ndcg_5)
                    vali_ndcgs_10.append(vali_ndcg_10)
                    vali_hrs_1.append(vali_hr_1)
                    vali_hrs_5.append(vali_hr_5)
                    vali_hrs_10.append(vali_hr_10)
                    vali_mrrs.append(vali_mrr)
                    vali_losses.append(vali_loss)

                    print("STEP %d  LOSS TRAIN: %.4f  LOSS VALI: %.4f  NDCG@5 VALI: %.4f  NDCG@10 VALI: %.4f  HR@1 VALI: %.4f  HR@5 VALI: %.4f  HR@10 VALI: %.4f  MRR VALI: %.4f" % (step, train_loss, vali_loss, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr))
                    if vali_mrrs[-1] > max(vali_mrrs[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.makedirs('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)

                    if len(vali_mrrs) > 2 and epoch > 0:
                        if (vali_mrrs[-1] < vali_mrrs[-2] and vali_mrrs[-2] < vali_mrrs[-3]):
                            early_stop = True
                        if (vali_mrrs[-1] - vali_mrrs[-2]) <= 0.001 and (vali_mrrs[-2] - vali_mrrs[-3]) <= 0.001:
                            early_stop = True

        # generate log
        if not os.path.exists('logs_{}/'.format(data_set)):
            os.makedirs('logs_{}/'.format(data_set))
        model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)

        with open('logs_{}/{}.pkl'.format(data_set, model_name), 'wb') as f:
            dump_tuple = (train_losses, vali_losses, vali_ndcgs_5, vali_ndcgs_10, vali_hrs_1, vali_hrs_5, vali_hrs_10, vali_mrrs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, model_name), 'w') as f:
            index = np.argmax(vali_mrrs)
            f.write('Result Validation NDCG@5: {}\n'.format(vali_ndcgs_5[index]))
            f.write('Result Validation NDCG@10: {}\n'.format(vali_ndcgs_10[index]))
            f.write('Result Validation HR@1: {}\n'.format(vali_hrs_1[index]))
            f.write('Result Validation HR@5: {}\n'.format(vali_hrs_5[index]))
            f.write('Result Validation HR@10: {}\n'.format(vali_hrs_10[index]))
            f.write('Result Validation MRR: {}\n'.format(vali_mrrs[index]))
        return vali_mrrs[index]
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set = sys.argv[3]

    if data_set == 'ccmr':
        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_CCMR + 'remap_movie_info_dict.pkl'
        user_fnum = 1 
        item_fnum = 5

        target_file_train = DATA_DIR_CCMR + 'target_38.txt'
        target_file_validation = DATA_DIR_CCMR + 'target_39_sample.txt'
        target_file_test = DATA_DIR_CCMR + 'target_40_sample.txt'
        
        user_seq_file_train = DATA_DIR_CCMR + 'train_user_hist_seq_38.txt'
        user_seq_file_validation = DATA_DIR_CCMR + 'validation_user_hist_seq_39_sample.txt'
        user_seq_file_test = DATA_DIR_CCMR + 'test_user_hist_seq_40_sample.txt'
        
        item_seq_file_train = DATA_DIR_CCMR + 'train_item_hist_seq_38.txt'
        item_seq_file_validation = DATA_DIR_CCMR + 'validation_item_hist_seq_39_sample.txt'
        item_seq_file_test = DATA_DIR_CCMR + 'test_item_hist_seq_40_sample.txt'
        
        # model parameter
        feature_size = FEAT_SIZE_CCMR
        max_time_len = MAX_LEN_CCMR
        dataset_size = 505530

    elif data_set == 'taobao':
        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_Taobao + 'item_feat_dict.pkl'
        user_fnum = 1 
        item_fnum = 2

        target_file_train = DATA_DIR_Taobao + 'target_6.txt'
        target_file_validation = DATA_DIR_Taobao + 'target_7_sample.txt'
        target_file_test = DATA_DIR_Taobao + 'target_8_sample.txt'

        user_seq_file_train = DATA_DIR_Taobao + 'train_user_hist_seq_6.txt'
        user_seq_file_validation = DATA_DIR_Taobao + 'validation_user_hist_seq_7_sample.txt'
        user_seq_file_test = DATA_DIR_Taobao + 'test_user_hist_seq_8_sample.txt'
        
        item_seq_file_train = DATA_DIR_Taobao + 'train_item_hist_seq_6.txt'
        item_seq_file_validation = DATA_DIR_Taobao + 'validation_item_hist_seq_7_sample.txt'
        item_seq_file_test = DATA_DIR_Taobao + 'test_item_hist_seq_8_sample.txt'

        # model parameter
        feature_size = FEAT_SIZE_Taobao
        max_time_len = MAX_LEN_Taobao
        dataset_size = 728919
    elif data_set == 'tmall':
        user_feat_dict_file = DATA_DIR_Tmall + 'user_feat_dict.pkl'
        item_feat_dict_file = DATA_DIR_Tmall + 'item_feat_dict.pkl'
        user_fnum = 3 
        item_fnum = 4

        target_file_train = DATA_DIR_Tmall + 'target_9.txt'
        target_file_validation = DATA_DIR_Tmall + 'target_10_sample.txt'
        target_file_test = DATA_DIR_Tmall + 'target_11_sample.txt'

        user_seq_file_train = DATA_DIR_Tmall + 'train_user_hist_seq_9.txt'
        user_seq_file_validation = DATA_DIR_Tmall + 'validation_user_hist_seq_10_sample.txt'
        user_seq_file_test = DATA_DIR_Tmall + 'test_user_hist_seq_11_sample.txt'
        
        item_seq_file_train = DATA_DIR_Tmall + 'train_item_hist_seq_9.txt'
        item_seq_file_validation = DATA_DIR_Tmall + 'validation_item_hist_seq_10_sample.txt'
        item_seq_file_test = DATA_DIR_Tmall + 'test_item_hist_seq_11_sample.txt'

        # model parameter
        feature_size = FEAT_SIZE_Tmall
        max_time_len = MAX_LEN_Tmall
        dataset_size = 222795
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    reg_lambdas = [1e-4, 5e-4, 5e-3]
    hyper_paras = [(100, 5e-4), (200, 1e-3)]
    
    vali_mrrs = []
    hyper_list = []

    for hyper in hyper_paras:
        train_batch_size, lr = hyper
        for reg_lambda in reg_lambdas:
            vali_mrr = train(data_set, target_file_train, target_file_validation, user_seq_file_train, user_seq_file_validation,
                    item_seq_file_train, item_seq_file_validation, model_type, train_batch_size, feature_size, 
                    EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, lr, reg_lambda, dataset_size, 
                    user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum)
            vali_mrrs.append(vali_mrr)
            hyper_list.append((train_batch_size, lr, reg_lambda))

    index = np.argmax(vali_mrrs)
    best_hyper = hyper_list[index]
    train_batch_size, lr, reg_lambda = best_hyper
    restore(data_set, target_file_test, user_seq_file_test, item_seq_file_test,
        model_type, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, 
        lr, reg_lambda, user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum)
