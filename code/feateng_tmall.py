import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt

RAW_DIR = '../score-data/Tmall/raw_data/'
FEATENG_DIR = '../score-data/Tmall/feateng/'
START_TIME = int(time.mktime(datetime.datetime.strptime('2015-5-1', "%Y-%m-%d").timetuple()))
TIME_DELTA = 15 * 24 * 3600

def join_user_profile(user_profile_file, behavior_file, joined_file):
    user_profile_dict = {}
    with open(user_profile_file, 'r') as f:
        for line in f:
            uid, aid, gid = line[:-1].split(',')
            user_profile_dict[uid] = ','.join([aid, gid])
    
    # join
    newlines = []
    with open(behavior_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            user_profile = user_profile_dict[uid]
            newlines.append(line[:-1] + ',' + user_profile + '\n')
    with open(joined_file, 'w') as f:
        f.writelines(newlines)

def preprocess_raw_data(raw_file, out_file, remap_dict_file, plt_file, user_feat_dict_file, item_feat_dict_file):
    time_idxs = []
    uid_set = set()
    iid_set = set()
    cid_set = set()
    sid_set = set()
    bid_set = set()
    aid_set = set()
    gid_set = set()
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, iid, cid, sid, bid, date_str, btypeid, aid, gid = line[:-1].split(',')
            uid_set.add(uid)
            iid_set.add(iid)
            cid_set.add(cid)
            sid_set.add(sid)
            bid_set.add(bid)
            aid_set.add(aid)
            gid_set.add(gid)
            date_str = '2015' + date_str
            time_int = int(time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple()))
            t_idx = (time_int - START_TIME) // TIME_DELTA
            time_idxs.append(t_idx)

    # remap
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)
    sid_list = list(sid_set)
    bid_list = list(bid_set)
    aid_list = list(aid_set)
    gid_list = list(gid_set)

    print('user num: {}'.format(len(uid_list)))
    print('item num: {}'.format(len(iid_list)))
    print('cate num: {}'.format(len(cid_list)))
    print('seller num: {}'.format(len(sid_list)))
    print('brand num: {}'.format(len(bid_list)))
    print('age num: {}'.format(len(aid_list)))
    print('gender num: {}'.format(len(gid_list)))
    
    remap_id = 1
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}
    sid_remap_dict = {}
    bid_remap_dict = {}
    aid_remap_dict = {}
    gid_remap_dict = {}

    for uid in uid_list:
        uid_remap_dict[uid] = str(remap_id)
        remap_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(remap_id)
        remap_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(remap_id)
        remap_id += 1
    for sid in sid_list:
        sid_remap_dict[sid] = str(remap_id)
        remap_id += 1
    for bid in bid_list:
        bid_remap_dict[bid] = str(remap_id)
        remap_id += 1
    for aid in aid_list:
        aid_remap_dict[aid] = str(remap_id)
        remap_id += 1
    for gid in gid_list:
        gid_remap_dict[gid] = str(remap_id)
        remap_id += 1
    print('feat size: {}'.format(remap_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
        pkl.dump(sid_remap_dict, f)
        pkl.dump(bid_remap_dict, f)
        pkl.dump(aid_remap_dict, f)
        pkl.dump(gid_remap_dict, f)
    print('remap ids completed')

    # remap file generate
    item_feat_dict = {}
    user_feat_dict = {}
    # for dummy user
    user_feat_dict['0'] = [0, 0]
    newlines = []
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            uid, iid, cid, sid, bid, time_stamp, btypeid, aid, gid = lines[i][:-1].split(',')
            uid_remap = uid_remap_dict[uid]
            iid_remap = iid_remap_dict[iid]
            cid_remap = cid_remap_dict[cid]
            sid_remap = sid_remap_dict[sid]
            bid_remap = bid_remap_dict[bid]
            aid_remap = aid_remap_dict[aid]
            gid_remap = gid_remap_dict[gid]
            t_idx = time_idxs[i]
            item_feat_dict[iid_remap] = [int(cid_remap), int(sid_remap), int(bid_remap)]
            user_feat_dict[uid_remap] = [int(aid_remap), int(gid_remap)]
            newlines.append(','.join([uid_remap, iid_remap, '_', str(t_idx)]) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('remaped file generated')


    with open(user_feat_dict_file, 'wb') as f:
        pkl.dump(user_feat_dict, f)
    print('user feat dict dump completed')
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)
    print('item feat dict dump completed')

    # plot distribution
    print('max t_idx: {}'.format(max(time_idxs)))
    print('min t_idx: {}'.format(min(time_idxs)))
    plt.hist(time_idxs, bins=range(max(time_idxs)+2))
    plt.savefig(plt_file)


if __name__ == "__main__":
    join_user_profile(RAW_DIR + 'user_info_format1.csv', RAW_DIR + 'user_log_format1.csv', FEATENG_DIR + 'joined_user_behavior.csv')
    preprocess_raw_data(FEATENG_DIR + 'joined_user_behavior.csv', FEATENG_DIR + 'remaped_user_behavior.csv', FEATENG_DIR + 'remap_dict.pkl', FEATENG_DIR + 'time_distri.png', FEATENG_DIR + 'user_feat_dict.pkl', FEATENG_DIR + 'item_feat_dict.pkl')