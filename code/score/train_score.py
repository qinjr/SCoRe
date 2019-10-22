import os
import tensorflow as tf
import sys
from graph_loader import *
from score import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 100
TRAIN_NEG_SAMPLE_NUM = 1
TEST_NEG_SAMPLE_NUM = 49

WORKER_N = 8

# for CCMR
OBJ_PER_TIME_SLICE_CCMR = 10
TIME_SLICE_NUM_CCMR = 41
START_TIME_CCMR = 0
FEAT_SIZE_CCMR = 1 + 4920695 + 190129 + (80171 + 1) + (213481 + 1) + (62 + 1) + (1043 + 1)
DATA_DIR_CCMR = '../../score-data/CCMR/feateng/'
USER_PER_COLLECTION_CCMR = 1000
ITEM_PER_COLLECTION_CCMR = 100
USER_NUM_CCMR = 4920695
ITEM_NUM_CCMR = 190129

# for Taobao
OBJ_PER_TIME_SLICE_Taobao = 10
TIME_SLICE_NUM_Taobao = 9
START_TIME_Taobao = 0
FEAT_SIZE_Taobao = 1 + 984080 + 4049268 + 9405
DATA_DIR_Taobao = '../../score-data/Taobao/feateng/'
USER_PER_COLLECTION_Taobao = 500
ITEM_PER_COLLECTION_Taobao = 500
USER_NUM_Taobao = 984080
ITEM_NUM_Taobao = 4049268

# for Tmall
OBJ_PER_TIME_SLICE_Tmall = 10
TIME_SLICE_NUM_Tmall = 12
START_TIME_Tmall = 0
FEAT_SIZE_Tmall = 1529672
DATA_DIR_Tmall = '../../score-data/Tmall/feateng/'
USER_PER_COLLECTION_Tmall = 200
ITEM_PER_COLLECTION_Tmall = 500
USER_NUM_Tmall = 424170
ITEM_NUM_Tmall = 1090390


def restore(data_set, target_file_test, graph_handler_params, start_time,
        pred_time_test, model_type, train_batch_size, feature_size, eb_dim, 
        hidden_size, max_time_len, obj_per_time_slice, lr, reg_lambda,
        user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum):
    print('restore begin')
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'RIA':
        model = RIA(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'RCA':
        model = RCA(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'SCORE_USER':
        model = SCORE_USER(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'SCORE_ITEM':
        model = SCORE_ITEM(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        _, _, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss = eval(model, sess, graph_handler_params, target_file_test, start_time, pred_time_test, reg_lambda)
        # p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        # rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print('RESTORE, LOSS TEST: %.4f  NDCG@5 TEST: %.4f  NDCG@10 TEST: %.4f  HR@1 TEST: %.4f  HR@5 TEST: %.4f  HR@10 TEST: %.4f  MRR TEST: %.4f' % (loss, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr))
        with open('logs_{}/{}_{}.test.result'.format(data_set, model_name, obj_per_time_slice), 'w') as f:
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

def eval(model, sess, graph_handler_params, target_file, start_time, pred_time, 
        reg_lambda):
    preds = []
    labels = []
    target_iids = []
    losses = []
    graph_loader = GraphLoader(graph_handler_params, EVAL_BATCH_SIZE, target_file, start_time, pred_time, WORKER_N, TEST_NEG_SAMPLE_NUM)
    t = time.time()
    for batch_data in graph_loader:
        pred, label, loss = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        losses.append(loss)
        target_iids += np.array(batch_data[5]).tolist()
    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr = get_ranking_quality(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return logloss, auc, ndcg_5, ndcg_10, hr_1, hr_5, hr_10, mrr, loss
    
def train(data_set, target_file_train, target_file_validation, graph_handler_params, start_time,
        pred_time_train, pred_time_validation, model_type, train_batch_size, feature_size, 
        eb_dim, hidden_size, max_time_len, obj_per_time_slice, lr, reg_lambda, dataset_size,
        user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum):
    graph_handler_params = graph_handler_params
    if model_type == 'SCORE':
        model = SCORE(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'RIA':
        model = RIA(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'RCA':
        model = RCA(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'SCORE_USER':
        model = SCORE_USER(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
    elif model_type == 'SCORE_ITEM':
        model = SCORE_ITEM(feature_size, eb_dim, hidden_size, max_time_len, obj_per_time_slice, user_fnum, item_fnum)
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
        _, _, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model, sess, graph_handler_params, target_file_validation, start_time, pred_time_validation, reg_lambda)
        
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
        for epoch in range(6):
            if early_stop:
                break
            graph_loader = GraphLoader(graph_handler_params, train_batch_size, target_file_train, start_time, pred_time_train, WORKER_N, TRAIN_NEG_SAMPLE_NUM)
            for batch_data in graph_loader:
                if early_stop:
                    break

                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []

                    _, _, vali_ndcg_5, vali_ndcg_10, vali_hr_1, vali_hr_5, vali_hr_10, vali_mrr, vali_loss = eval(model, sess, graph_handler_params, target_file_validation, start_time, pred_time_validation, reg_lambda)

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

        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_CCMR, 'ccmr_2hop', OBJ_PER_TIME_SLICE_CCMR, \
                                USER_NUM_CCMR, ITEM_NUM_CCMR, START_TIME_CCMR, \
                                USER_PER_COLLECTION_CCMR, \
                                ITEM_PER_COLLECTION_CCMR, 'rs', \
                                user_feat_dict_file, item_feat_dict_file, \
                                user_fnum, item_fnum]
        target_file_train = DATA_DIR_CCMR + 'target_38.txt'
        target_file_validation = DATA_DIR_CCMR + 'target_39_sample.txt'
        target_file_test = DATA_DIR_CCMR + 'target_40_sample.txt'

        start_time = START_TIME_CCMR
        pred_time_train = 38
        pred_time_validation = 39
        pred_time_test = 40

        # model parameter
        feature_size = FEAT_SIZE_CCMR
        max_time_len = TIME_SLICE_NUM_CCMR - START_TIME_CCMR - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_CCMR
        dataset_size = 505530
    elif data_set == 'taobao':
        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_Taobao + 'item_feat_dict.pkl'
        user_fnum = 1 
        item_fnum = 2

        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_Taobao, 'taobao_2hop', OBJ_PER_TIME_SLICE_Taobao, \
                                USER_NUM_Taobao, ITEM_NUM_Taobao, START_TIME_Taobao, \
                                USER_PER_COLLECTION_Taobao, ITEM_PER_COLLECTION_Taobao, 'rs', \
                                user_feat_dict_file, item_feat_dict_file, \
                                user_fnum, item_fnum]

        target_file_train = DATA_DIR_Taobao + 'target_6.txt'
        target_file_validation = DATA_DIR_Taobao + 'target_7_sample.txt'
        target_file_test = DATA_DIR_Taobao + 'target_8_sample.txt'
        
        start_time = START_TIME_Taobao
        pred_time_train = 6
        pred_time_validation = 7
        pred_time_test = 8
    
        # model parameter
        feature_size = FEAT_SIZE_Taobao
        max_time_len = TIME_SLICE_NUM_Taobao - START_TIME_Taobao - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_Taobao
        dataset_size = 728919
    elif data_set == 'tmall':
        user_feat_dict_file = DATA_DIR_Tmall + 'user_feat_dict.pkl'
        item_feat_dict_file = DATA_DIR_Tmall + 'item_feat_dict.pkl'
        user_fnum = 3 
        item_fnum = 4

        # graph loader
        graph_handler_params = [TIME_SLICE_NUM_Tmall, 'tmall_2hop', OBJ_PER_TIME_SLICE_Tmall, \
                                USER_NUM_Tmall, ITEM_NUM_Tmall, START_TIME_Tmall, \
                                USER_PER_COLLECTION_Tmall, ITEM_PER_COLLECTION_Tmall, 'rs', \
                                user_feat_dict_file, item_feat_dict_file, \
                                user_fnum, item_fnum]
        target_file_train = DATA_DIR_Tmall + 'target_9.txt'
        target_file_validation = DATA_DIR_Tmall + 'target_10_sample.txt'
        target_file_test = DATA_DIR_Tmall + 'target_11_sample.txt'

        start_time = START_TIME_Tmall
        pred_time_train = 9
        pred_time_validation = 10
        pred_time_test = 11
        
        # model parameter
        feature_size = FEAT_SIZE_Tmall
        max_time_len = TIME_SLICE_NUM_Tmall - START_TIME_Tmall - 1
        obj_per_time_slice = OBJ_PER_TIME_SLICE_Tmall
        dataset_size = 222795
    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    # TRAINING PROCESS
    reg_lambdas = [1e-4, 5e-4, 5e-3]
    hyper_paras = [(100, 5e-4), (200, 1e-3)]

    vali_mrrs = []
    hyper_list = []

    for hyper in hyper_paras:
        train_batch_size, lr = hyper
        for reg_lambda in reg_lambdas:
            vali_mrr = train(data_set, target_file_train, target_file_validation, graph_handler_params, start_time,
                    pred_time_train, pred_time_validation, model_type, train_batch_size, feature_size, 
                    EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, obj_per_time_slice, lr, reg_lambda, dataset_size,
                    user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum)
            vali_mrrs.append(vali_mrr)
            hyper_list.append((train_batch_size, lr, reg_lambda))
    
    # TEST
    index = np.argmax(vali_mrrs)
    best_hyper = hyper_list[index]
    train_batch_size, lr, reg_lambda = best_hyper
    restore(data_set, target_file_test, graph_handler_params, start_time,
            pred_time_test, model_type, train_batch_size, feature_size, 
            EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, obj_per_time_slice, 
            lr, reg_lambda, user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum)

    # TESTING PROCESS
    # obj_per_time_slice_ops = [5, 10, 15, 20]
    # reg_lambdas = [1e-4, 1e-5]
    # train_batch_size, lr = 100, 5e-4
    # for reg_lambda in reg_lambdas:
    #     for obj_per_time_slice in obj_per_time_slice_ops:
    #         graph_handler_params[2] = obj_per_time_slice
    #         print('REG_LAMBDA is:{}, OBJ_PER_TIME_SLICE is {}:'.format(reg_lambda, obj_per_time_slice))
    #         restore(data_set, target_file_test, graph_handler_params, start_time,
    #                 pred_time_test, model_type, train_batch_size, feature_size, 
    #                 EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, obj_per_time_slice, 
    #                 lr, reg_lambda, user_feat_dict_file, item_feat_dict_file, user_fnum, item_fnum)

