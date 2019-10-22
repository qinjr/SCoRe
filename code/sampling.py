import random
import sys

# CCMR dataset parameters
DATA_DIR_CCMR = '../score-data/CCMR/feateng/'

# Taobao dataset parameters
DATA_DIR_Taobao = '../score-data/Taobao/feateng/'

# Tmall dataset parameters
DATA_DIR_Tmall = '../score-data/Tmall/feateng/'

random.seed(11)

def sample_files(target_file, 
                user_seq_file, 
                item_seq_file, 
                sample_target_file, 
                sample_user_seq_file, 
                sample_item_seq_file,
                sample_factor):
    print('sampling begin')
    target_lines = open(target_file).readlines()
    user_seq_lines = open(user_seq_file).readlines()
    item_seq_lines = open(item_seq_file).readlines()

    sample_target_lines = []
    sample_user_seq_lines = []
    sample_item_seq_lines = []
    
    length = len(target_lines)
    for i in range(length):
        rand_int = random.randint(1, sample_factor)
        if rand_int == 1:
            sample_target_lines.append(target_lines[i])
            sample_user_seq_lines.append(user_seq_lines[i])
            sample_item_seq_lines.append(item_seq_lines[i])

    with open(sample_target_file, 'w') as f:
        f.writelines(sample_target_lines)
    with open(sample_user_seq_file, 'w') as f:
        f.writelines(sample_user_seq_lines)
    with open(sample_item_seq_file, 'w') as f:
        f.writelines(sample_item_seq_lines)
    print('sampling end')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PLEASE INPUT [DATASET]")
        sys.exit(0)
    dataset = sys.argv[1]
    if dataset == 'ccmr':
        # CCMR
        sample_files(DATA_DIR_CCMR + 'target_39.txt', 
                    DATA_DIR_CCMR + 'validation_user_hist_seq_39.txt', 
                    DATA_DIR_CCMR + 'validation_item_hist_seq_39.txt',
                    DATA_DIR_CCMR + 'target_39_sample.txt', 
                    DATA_DIR_CCMR + 'validation_user_hist_seq_39_sample.txt', 
                    DATA_DIR_CCMR + 'validation_item_hist_seq_39_sample.txt', 
                    100)
        sample_files(DATA_DIR_CCMR + 'target_40.txt', 
                    DATA_DIR_CCMR + 'test_user_hist_seq_40.txt', 
                    DATA_DIR_CCMR + 'test_item_hist_seq_40.txt', 
                    DATA_DIR_CCMR + 'target_40_sample.txt', 
                    DATA_DIR_CCMR + 'test_user_hist_seq_40_sample.txt', 
                    DATA_DIR_CCMR + 'test_item_hist_seq_40_sample.txt', 
                    20)
    elif dataset == 'taobao':
        # Taobao
        sample_files(DATA_DIR_Taobao + 'target_7.txt', 
                    DATA_DIR_Taobao + 'validation_user_hist_seq_7.txt', 
                    DATA_DIR_Taobao + 'validation_item_hist_seq_7.txt', 
                    DATA_DIR_Taobao + 'target_7_sample.txt', 
                    DATA_DIR_Taobao + 'validation_user_hist_seq_7_sample.txt', 
                    DATA_DIR_Taobao + 'validation_item_hist_seq_7_sample.txt', 
                    40)
        sample_files(DATA_DIR_Taobao + 'target_8.txt', 
                    DATA_DIR_Taobao + 'test_user_hist_seq_8.txt', 
                    DATA_DIR_Taobao + 'test_item_hist_seq_8.txt', 
                    DATA_DIR_Taobao + 'target_8_sample.txt', 
                    DATA_DIR_Taobao + 'test_user_hist_seq_8_sample.txt', 
                    DATA_DIR_Taobao + 'test_item_hist_seq_8_sample.txt', 
                    20)
    elif dataset == 'tmall':
        # Tmall
        sample_files(DATA_DIR_Tmall + 'target_10.txt', 
                    DATA_DIR_Tmall + 'validation_user_hist_seq_10.txt', 
                    DATA_DIR_Tmall + 'validation_item_hist_seq_10.txt', 
                    DATA_DIR_Tmall + 'target_10_sample.txt', 
                    DATA_DIR_Tmall + 'validation_user_hist_seq_10_sample.txt',
                    DATA_DIR_Tmall + 'validation_item_hist_seq_10_sample.txt', 
                    12)
        sample_files(DATA_DIR_Tmall + 'target_11.txt', 
                    DATA_DIR_Tmall + 'test_user_hist_seq_11.txt', 
                    DATA_DIR_Tmall + 'test_item_hist_seq_11.txt', 
                    DATA_DIR_Tmall + 'target_11_sample.txt', 
                    DATA_DIR_Tmall + 'test_user_hist_seq_11_sample.txt',
                    DATA_DIR_Tmall + 'test_item_hist_seq_11_sample.txt', 
                    4)
    else:
        print('WRONG DATASET: {}'.format(dataset))

