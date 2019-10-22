import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt

RAW_DIR = '../score-data/Taobao/raw_data/'
FEATENG_DIR = '../score-data/Taobao/feateng/'

START_TIME = int(time.mktime(datetime.datetime.strptime('2017-11-25', "%Y-%m-%d").timetuple()))
END_TIME = int(time.mktime(datetime.datetime.strptime('2017-12-04', "%Y-%m-%d").timetuple()))
TIME_DELTA = 24 * 3600

# filtering and time to index and remap ids and plot distribution
def preprocess_raw_data(in_file, out_file_filtered, out_file_remaped, plt_file, remap_dict_file, item_feat_dict_file):
    newlines = []
    newlines_filtered = []
    time_idxs = []
    uid_set = set()
    iid_set = set()
    cid_set = set()

    with open(in_file, 'r') as f:
        for line in f:
            uid, iid, cid, btype, time = line[:-1].split(',')
            if int(time) < START_TIME or int(time) > END_TIME:
                continue
            if btype != 'pv':
                continue
            newlines_filtered.append(','.join([uid, iid, cid, time]) + '\n')
            time_idxs.append((int(time) - START_TIME) // TIME_DELTA)
            uid_set.add(uid)
            iid_set.add(iid)
            cid_set.add(cid)
    with open(out_file_filtered, 'w') as f:
        f.writelines(newlines_filtered)

    # remap dict
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)

    print('user num: {}'.format(len(uid_list)))
    print('item num: {}'.format(len(iid_list)))
    print('cate num: {}'.format(len(cid_list)))

    remap_id = 1
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}

    for uid in uid_list:
        uid_remap_dict[uid] = str(remap_id)
        remap_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(remap_id)
        remap_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(remap_id)
        remap_id += 1
    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
    print('remap ids completed')
    
    # remap file generate
    item_feat_dict = {}
    with open(in_file, 'r') as f:
        for line in f:
            uid, iid, cid, btype, time = line[:-1].split(',')
            if int(time) < START_TIME or int(time) > END_TIME:
                continue
            if btype != 'pv':
                continue
            uid_remap = uid_remap_dict[uid]
            iid_remap = iid_remap_dict[iid]
            cid_remap = cid_remap_dict[cid]
            time_idx = str((int(time) - START_TIME) // TIME_DELTA)
            item_feat_dict[iid_remap] = [int(cid_remap)]
            newlines.append(','.join([uid_remap, iid_remap, cid_remap, time_idx]) + '\n')
    with open(out_file_remaped, 'w') as f:
        f.writelines(newlines)
    print('remaped file generated')
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)
    print('item feat dict dump completed')

    # plot distribution
    print('max t_idx: {}'.format(max(time_idxs)))
    print('min t_idx: {}'.format(min(time_idxs)))
    plt.hist(time_idxs, bins=range(max(time_idxs)+2))
    plt.savefig(plt_file)


if __name__ == "__main__":
    preprocess_raw_data(RAW_DIR + 'UserBehavior.csv', FEATENG_DIR + 'filtered_user_behavior.txt',
                        FEATENG_DIR + 'remaped_user_behavior.txt', FEATENG_DIR + 'time_idx_distri.png', 
                        FEATENG_DIR + 'remap_dict.pkl', FEATENG_DIR + 'item_feat_dict.pkl')