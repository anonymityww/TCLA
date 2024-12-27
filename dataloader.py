import torch
import torch.utils.data as Data
from tqdm import tqdm
import random
import math

def create_sub_seqs(seq, max_seqs_len):
    seqs_len = len(seq)
    index = random.randint(0, seqs_len - 1)
    sub_seqs_len = random.randint(1, max_seqs_len)
    seq = seq[index:index + sub_seqs_len]
    return seq

def random_modify(raw_data, item_num, modified_max_seq_len, max_insert_num, plist):
    seq_len = len(raw_data)
    random_modified_data = []
    l1_ground_truth = torch.zeros([modified_max_seq_len], dtype=torch.long)
    l2_ground_truth = torch.zeros([modified_max_seq_len, max_insert_num], dtype=torch.long)
    available_item = set(i for i in range(1, item_num - 2))
    for item_id in raw_data:
        available_item.discard(item_id)

    modified_index = 0
    del_seqs = []
    raw_index = 0

    while raw_index < len(raw_data):
        decision = random.random()
        if decision <= plist[0]:
            random_modified_data.append(raw_data[raw_index])
            modified_index += 1
            raw_index += 1
        elif plist[0] < decision <= plist[1] and seq_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
            del_seqs.insert(0, raw_data[raw_index])
            seq_len -= 1
            raw_index += 1
            decision = random.random()

            while plist[0] < decision <= plist[1] and seq_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
                del_seqs.insert(0, raw_data[raw_index])
                seq_len -= 1
                raw_index += 1
                decision = random.random()

            if len(del_seqs) < max_insert_num:
                del_seqs.append(item_num - 2)
                del_seqs = del_seqs + [0] * (max_insert_num - len(del_seqs))

            del_seqs = torch.tensor(del_seqs, dtype=torch.long)
            l2_ground_truth[modified_index] = del_seqs
            l1_ground_truth[modified_index] = 2
            random_modified_data.append(raw_data[raw_index])
            modified_index += 1
            raw_index += 1
            del_seqs = []

        elif decision > plist[1] and len(available_item) != 0 and seq_len < modified_max_seq_len:
            insert_item = random.sample(available_item, 1)
            random_modified_data.append(insert_item[0])
            available_item.remove(insert_item[0])
            l1_ground_truth[modified_index] = 1
            modified_index += 1
            seq_len += 1

    random_modified_data = random_modified_data + [0] * (modified_max_seq_len - len(random_modified_data))
    random_modified_data = torch.tensor(random_modified_data, dtype=torch.long)
    return random_modified_data, l1_ground_truth, l2_ground_truth


class TrainDataset(Data.Dataset):
    def __init__(self, dir_path, item_num, max_seq_len, modified_max_seq_len, max_insert_num, mask_prob, plist):
        super(TrainDataset, self).__init__()
        self.raw_data = []
        self.item_num = item_num
        self.max_seqs_len = max_seq_len
        self.modified_max_seqs_len = modified_max_seq_len
        self.all_items = set(range(1, self.item_num-2))
        self.max_insert_num = max_insert_num
        self.mask_prob = mask_prob
        self.plist = plist

        with open(dir_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                seq = seq[-self.max_seqs_len:]
                for i in range(len(seq)):
                    if len(seq[:i+1]) >= 2:
                      self.raw_data.append(seq[:i+1])

    def augment(self, item_seq, item_seq_len):
        if item_seq_len > 1:
            switch = random.sample(range(3), k=1)
        else:
            switch = [3]
            aug_seq = item_seq

        if switch[0] == 0:
            aug_seq = self.item_crop(item_seq, item_seq_len)
        elif switch[0] == 1:
            aug_seq = self.item_mask(item_seq, item_seq_len)
        elif switch[0] == 2:
            aug_seq = self.item_reorder(item_seq, item_seq_len)

        return aug_seq

    def item_crop(self, item_seq, item_seq_len, eta=0.6):
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = torch.zeros(item_seq.shape[0], dtype=torch.long)
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[:num_left] = item_seq[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[:num_left] = item_seq[crop_begin:]
        return croped_item_seq

    def item_mask(self, item_seq, item_seq_len, gamma=0.3):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.clone()
        masked_item_seq[mask_index] = self.item_num-1  # token 0 has been used for semantic masking（0 is pad，item_num-1 is mask）
        return masked_item_seq

    def item_reorder(self, item_seq, item_seq_len, beta=0.6):
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.clone()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return reordered_item_seq


    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        seq = self.raw_data[index].copy()
        item_set = set(seq)

        random_modified_seq, l1_ground_truth, l2_ground_truth = random_modify(seq, self.item_num, self.modified_max_seqs_len, self.max_insert_num, self.plist)

        input_seqs_ids = seq[:-1] + [0] * (self.modified_max_seqs_len - len(seq[:-1]))
        input_seqs_ids = torch.tensor(input_seqs_ids).long()

        target = seq[-1]
        target = torch.tensor(target).long()

        negative = random.sample(self.all_items-item_set, 1)[0]
        nagative = torch.tensor(negative).long()

        rand_aug_seq = self.augment(input_seqs_ids, len(seq)-1)

        insert_seqs = torch.zeros([self.modified_max_seqs_len, self.max_insert_num - 1], dtype=torch.long)
        insert_seqs[:] = l2_ground_truth[:, :-1]

        return input_seqs_ids, target, nagative, rand_aug_seq, random_modified_seq, l1_ground_truth, l2_ground_truth, insert_seqs


class ValidDataset(Data.Dataset):
    def __init__(self, data_path, neg_path, item_num, modified_max_seq_len):
        super(ValidDataset, self).__init__()
        self.data = []
        self.neg_data = []
        self.item_num = item_num
        self.modified_max_seq_len = modified_max_seq_len

        with open(data_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.data.append(seq)

        with open(neg_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip.split(',')
                seq = [int(i) for i in seq]
                self.neg_data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index].copy()

        input_seqs_ids = seq[:-1] + [0] * (self.modified_max_seq_len - len(seq[:-1]))
        input_seqs_ids = torch.tensor(input_seqs_ids).long()

        target = seq[-1]
        target = torch.tensor(target, dtype=torch.long)

        negatives = self.neg_data[index]
        neg_num = len(negatives)
        neg_num = torch.tensor(neg_num).long()
        negatives = torch.tensor(negatives).long()

        return input_seqs_ids, target, negatives, neg_num

class TestDataset(Data.Dataset):
    def __init__(self, data_path, neg_path, item_num, modified_max_seq_len):
        super(TestDataset, self).__init__()
        self.data = []
        self.neg_data = []
        self.item_num = item_num
        self.modified_max_seq_len = modified_max_seq_len

        with open(data_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.data.append(seq)

        with open(neg_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.neg_data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index].copy()

        input_seqs_ids = seq[:-1] + [0] * (self.modified_max_seq_len - len(seq[:-1]))
        input_seqs_ids = torch.tensor(input_seqs_ids).long()

        target = seq[-1]
        target = torch.tensor(target, dtype=torch.long)

        negatives = self.neg_data[index]
        neg_num = len(negatives)
        neg_num = torch.tensor(neg_num).long()
        negatives = torch.tensor(negatives).long()

        return input_seqs_ids, target, negatives, neg_num
