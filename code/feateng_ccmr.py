import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt

RAW_DIR = '../score-data/CCMR/raw_data/'
FEATENG_DIR = '../score-data/CCMR/feateng/'

USER_NUM = 4920695
ITEM_NUM = 190129
DIRECTOR_NUM = 80171 + 1 #+1 no director
ACTOR_NUM = 213481 + 1 #+1 no actor
GENRE_NUM = 62 + 1 #+1 no genre
NATION_NUM = 1043 + 1 #+1 no nation


TIME_DELTA = 90
SECONDS_PER_DAY = 24 * 3600

START_TIME = 1116432000#int(time.mktime(datetime.datetime.strptime('2011-01-01', "%Y-%m-%d").timetuple()))

def pos_neg_split(in_file, pos_file, neg_file):
    pos_lines = []
    neg_lines = []
    with open(in_file, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            else:
                if line.split(',')[2] == '5' or line.split(',')[2] == '4':
                    pos_lines.append(line)
                else:
                    neg_lines.append(line)
    
    print('pos sample: {}'.format(len(pos_lines)))
    print('neg sample: {}'.format(len(neg_lines)))
    with open(pos_file, 'w') as f:
        f.writelines(pos_lines)
    with open(neg_file, 'w') as f:
        f.writelines(neg_lines)

# filter and transfer to time index
def time_filter(in_file, out_file):
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid, iid, rating, time_str = line[:-1].split(',')
            time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
            if time_int >= START_TIME:
                newlines.append(','.join([uid, iid, rating, str(int((time_int - START_TIME) / (SECONDS_PER_DAY * TIME_DELTA)))]) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('time filtering completed')


def time_distri(in_file, plt_file):
    times = []
    with open(in_file, 'r') as f:
        for line in f:
            time = int(line[:-1].split(',')[-1])
            times.append(time)
    print('max time idx: {}'.format(max(times)))

    plt.hist(times, bins=range(max(times)+1))
    plt.savefig(plt_file)

def movie_feat_info(in_file):
    field_dict = {
        'director': [],
        'actor': [],
        'genre': [],
        'nation': []
    }
    director_width, actor_width, genre_width, nation_width = 0, 0, 0, 0
    with open(in_file, 'r') as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            _, directors, actors, genres, nations, __ = line.split(',')
            if directors != '':
                director_list = directors.split(';')
                field_dict['director'] += director_list
                if len(director_list) > director_width:
                    director_width = len(director_list)
            if actors != '':    
                actor_list = actors.split(';')
                field_dict['actor'] += actor_list
                if len(actor_list) > actor_width:
                    actor_width = len(actor_list)
            if genres != '':    
                genre_list = genres.split(';')
                field_dict['genre'] += genre_list
                if len(genre_list) > genre_width:
                    genre_width = len(genre_list)
            if nations != '':    
                nation_list = nations.split(';')
                field_dict['nation'] += nation_list
                if len(nation_list) > nation_width:
                    nation_width = len(nation_list)
    
    for key in field_dict:
        field_dict[key] = set(field_dict[key])
        field_dict[key] = set(map(int, field_dict[key]))
        print(key, len(field_dict[key]))
        print(key, max(field_dict[key]))
        print(key, min(field_dict[key]))
    print(director_width, actor_width, genre_width, nation_width)

# def time2idx_(time_str):
#     time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
#     return str(int((time_int - START_TIME) / (SECONDS_PER_DAY * TIME_DELTA)))

def time2idx(in_file, out_file):
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            uid, iid, rating, time_str = line[:-1].split(',')
            time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
            newline = ','.join([uid, iid, rating, str(int((time_int - START_TIME) / (SECONDS_PER_DAY * TIME_DELTA)))]) + '\n'
            newlines.append(newline)
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('time2idx completed')

def remap_ids(rating_file, new_rating_file, movie_info_file = None, new_movie_info_file = None):
    # remap rating_file
    new_rating_lines = []
    with open(rating_file, 'r') as f:
        for line in f:
            uid, iid, rating, time = line[:-1].split(',')
            newline = ','.join([str(int(uid) + 1), str(int(iid) + 1 + USER_NUM), rating, time]) + '\n'
            new_rating_lines.append(newline)
    with open(new_rating_file, 'w') as f:
        f.writelines(new_rating_lines)
    print('remap rating file completed')

    # remap movie info
    if movie_info_file != None:
        movie_info_dict = {} #iid(str): [did, aid, gid, nid](int)
        with open(movie_info_file, 'r') as f:
            for line in f:
                iid, directors, actors, genres, nations, _ = line[:-1].split(',')
                iid = str(int(iid) + 1 + USER_NUM)
                if directors == '':
                    did = 1 + USER_NUM + ITEM_NUM + DIRECTOR_NUM - 1
                else:
                    did = 1 + int(directors.split(';')[0]) + USER_NUM + ITEM_NUM
                if actors == '':
                    aid = 1 + USER_NUM + ITEM_NUM + DIRECTOR_NUM + ACTOR_NUM - 1
                else:
                    aid = 1 + int(actors.split(';')[0]) + USER_NUM + ITEM_NUM + DIRECTOR_NUM
                if genres == '':
                    gid = 1 + USER_NUM + ITEM_NUM + DIRECTOR_NUM + ACTOR_NUM + GENRE_NUM - 1
                else:
                    gid = 1 + int(genres.split(';')[0]) + USER_NUM + ITEM_NUM + DIRECTOR_NUM + ACTOR_NUM
                if nations == '':
                    nid = 1 + USER_NUM + ITEM_NUM + DIRECTOR_NUM + ACTOR_NUM + GENRE_NUM + NATION_NUM - 1
                else:
                    nid = 1 + int(nations.split(';')[0]) + USER_NUM + ITEM_NUM + DIRECTOR_NUM + ACTOR_NUM + GENRE_NUM
                
                if iid not in movie_info_dict:
                    movie_info_dict[iid] = [did, aid, gid, nid]
        with open(new_movie_info_file, 'wb') as f:
            pkl.dump(movie_info_dict, f)
        print('remap movie info completed')

def gen_user_neg(in_file, out_file):
    user_neg_dict = {} #uid(str):[iid, ...](str)
    with open(in_file, 'r') as f:
        for line in f:
            uid, iid, _, __ = line.split(',')
            if uid not in user_neg_dict:
                user_neg_dict[uid] = [iid]
            else:
                user_neg_dict[uid].append(iid)
    with open(out_file, 'wb') as f:
        pkl.dump(user_neg_dict, f)

def simplify_data(in_file, out_file):
    newlines = []
    with open(in_file, 'r') as f:
        for line in f:
            newlines.append(','.join(line.split(',')[:4]) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('simplify finished')

if __name__ == "__main__":
    simplify_data(RAW_DIR + 'rating_logs.csv', FEATENG_DIR + 'rating_logs.csv')
    pos_neg_split(RAW_DIR + 'rating_logs.csv', FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'rating_neg.csv')
    time2idx(FEATENG_DIR + 'rating_pos.csv', FEATENG_DIR + 'rating_pos_idx.csv')
    time2idx(FEATENG_DIR + 'rating_neg.csv', FEATENG_DIR + 'rating_neg_idx.csv')
    time_distri(FEATENG_DIR + 'rating_pos_idx.csv', FEATENG_DIR + 'time_distri.png')
    movie_feat_info(RAW_DIR + 'movie_info_colname.csv')
    remap_ids(FEATENG_DIR + 'rating_pos_idx.csv', FEATENG_DIR + 'remap_rating_pos_idx.csv', movie_info_file = RAW_DIR + 'movie_info.csv', new_movie_info_file = FEATENG_DIR + 'remap_movie_info_dict.pkl')
    remap_ids(FEATENG_DIR + 'rating_neg_idx.csv', FEATENG_DIR + 'remap_rating_neg_idx.csv')
    gen_user_neg(FEATENG_DIR + 'remap_rating_neg_idx.csv', FEATENG_DIR + 'user_neg_dict.pkl')