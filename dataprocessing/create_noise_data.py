'''
We use this to create simulated dataset
'''

from tqdm import tqdm
import torch
import random
import numpy as np
import time

def raw_data_rand_modify(raw_data, item_num, modified_max_seqs_len, max_insert_num):

    current_seqs_len = len(raw_data)

    random_modified_data = []

    available_item_pool = set(i for i in range(1, item_num - 2))

    for item_id in raw_data:

        available_item_pool.discard(item_id)

    modified_index = 0

    del_seqs = []

    raw_index = 0

    while raw_index < len(raw_data):

        decision = random.random()

        if decision <= 0.4:
            random_modified_data.append(raw_data[raw_index])
            modified_index += 1
            raw_index += 1

        elif 0.4 < decision <= 0.7 and current_seqs_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
            del_seqs.insert(0, raw_data[raw_index])
            current_seqs_len -= 1
            raw_index += 1

            decision = random.random()

            while 0.4 < decision <= 0.7 and current_seqs_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
                del_seqs.insert(0, raw_data[raw_index])
                current_seqs_len -= 1
                raw_index += 1

                decision = random.random()

            if len(del_seqs) < max_insert_num:
                del_seqs.append(item_num - 2)
                del_seqs = del_seqs + [0] * (max_insert_num - len(del_seqs))

            del_seqs = torch.tensor(del_seqs, dtype=torch.long)
            random_modified_data.append(raw_data[raw_index])
            modified_index += 1
            raw_index += 1
            del_seqs = []

        elif decision > 0.7 and len(available_item_pool) != 0 and current_seqs_len < modified_max_seqs_len:

            insert_item = random.sample(available_item_pool, 1)
            random_modified_data.append(insert_item[0])
            available_item_pool.remove(insert_item[0])
            modified_index += 1
            current_seqs_len += 1

    random_modified_data = random_modified_data + [0] * (modified_max_seqs_len - len(random_modified_data))
    random_modified_data = torch.tensor(random_modified_data, dtype=torch.long)

    return random_modified_data

def random_modify_test_data(input_file, output_file, item_num, modified_max_seqs_len, max_insert_num):
    start = time.time()
    with open(input_file, 'r', encoding='utf-8') as f1, open(output_file, 'a+', encoding='utf-8') as f2:
        rowID = 0
        for line in tqdm(f1):
            raw_data = line.strip().split(',')
            raw_data = [int(i) for i in raw_data]
            valid_data = raw_data[-2]
            test_data = raw_data[-1]
            modifiable_seqs = raw_data[:-2]
            random_modified_test_data = raw_data_rand_modify(modifiable_seqs, item_num, modified_max_seqs_len-2,
                                                                      max_insert_num)
            random_modified_test_data = random_modified_test_data.tolist()
            for k,v in enumerate(random_modified_test_data):
                if v == 0:
                    del random_modified_test_data[k:]
                    break
            random_modified_test_data.append(valid_data)
            random_modified_test_data.append(test_data)
            f2.write(','.join(map(str, random_modified_test_data))+'\n')

            if rowID % 100 == 0:
                now = time.time()
                print('done:{},time {}'.format(rowID, now-start))

            rowID += 1
    return

def train_valid_create(test_file, train_file, valid_file):
    with open(test_file, 'r', encoding='utf-8') as f,open(train_file, 'a+', encoding='utf-8') as w1,open(valid_file, 'a+', encoding='utf-8') as w2:
        for line in tqdm(f):
            item = line.strip().split(',')
            w1.write(','.join(map(str, item[:-2]))+'\n')
            w2.write(','.join(map(str, item[:-1]))+'\n')

if __name__ == '__main__':
    random_modify_test_data(r'./Sports/test.dat', r'./Sports/noisy_433_test.dat', 18360, 60, 5)
    train_valid_create(r'./Sports/noisy_433_test.dat',
                       r'./Sports/noisy_433_train.dat',
                       r'./Sports/noisy_433_valid.dat')
