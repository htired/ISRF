import math
import json
import numpy as np
import torch
import random
import datetime
from util.rouge import rouge
from util.bleu import compute_bleu
from model.templates import exp_templates, seq_templates, topn_templates


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,51240314116    
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


class ExpDataLoader:
    def __init__(self, data_dir):
        with open(data_dir + 'explanation.json', 'r') as f:
            self.exp_data = json.load(f)

        self.train = self.exp_data['train']
        self.valid = self.exp_data['val']
        self.test = self.exp_data['test']


class SeqDataLoader:
    def __init__(self, data_dir):
        self.user2items_positive = {}

        with open(data_dir + 'sequential.txt', 'r') as f:
            for line in f.readlines():
                user, items = line.strip().split(' ', 1)
                self.user2items_positive[int(user)] = items.split(' ')

        self.user2items_negative = {}
        with open(data_dir + 'negative.txt', 'r') as f:
            for line in f.readlines():
                user, items = line.strip().split(' ', 1)
                self.user2items_negative[int(user)] = items.split(' ')

        with open(data_dir + 'datamaps.json', 'r') as f:
            datamaps = json.load(f)
        self.id2user = datamaps['id2user']
        self.id2item = datamaps['id2item']


def compute_whole_word_id(task_id, seq_batch, tokenizer, max_len, user_num, alpha=1.0, decay_type="exp"):
    whole_word_ids = []
    whole_word_ids_users = []
    for seq_idx, seq in enumerate(seq_batch):
        token_list = tokenizer.tokenize(seq)
        start_indices = []
        
        for idx, token in enumerate(token_list):
            if token == '_':
                start_indices.append(idx - 1) 
        end_indices = []
        for start in start_indices:
            mover = start + 2  # user/item _ xx
            while mover < len(token_list) and token_list[mover].isdigit():
                mover += 1
            end_indices.append(mover)

        whole_word_id = [0] * len(token_list) 
        whole_word_id_user = [0] * len(token_list) 
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            # sequential recommendation task
            if task_id == 1:
                whole_word_id[start:end] = [i + 1] * (end - start)  # leave 0 as padding token
                if start+2 == end:
                    continue
                if token_list[start] == '▁user':
                    idx = int(''.join(token_list[start + 2:end]))
                    whole_word_id_user[start:end] = [idx] * (end - start)   
            # other recommendation tasks
            else:
                if start+2 == end:
                    continue
                idx = int(''.join(token_list[start + 2:end]))
                # ['▁user', '_', '122', '52', '▁item', '_', '86', '68']
                if token_list[start] == '▁user':
                    whole_word_id[start:end] = [idx] * (end - start)
                else:
                    whole_word_id[start:end] = [idx + user_num] * (end - start)

        whole_word_ids.append(whole_word_id)
        whole_word_ids_users.append(whole_word_id_user)
    padded_whole_word_ids = []
    for whole_word_id in whole_word_ids:
        padded_whole_word_ids.append(whole_word_id + [0] * (max_len - len(whole_word_id)))
    padded_whole_word_ids_user = []
    for whole_word_id_user in whole_word_ids_users:
        padded_whole_word_ids_user.append(whole_word_id_user + [0] * (max_len - len(whole_word_id_user)))

    if task_id == 1:
        return padded_whole_word_ids, padded_whole_word_ids_user
    return padded_whole_word_ids, None
class ExpSampler:
    def __init__(self, exp_data):
        self.task_id = 0
        self.exp_data = exp_data
        self.sample_num = len(self.exp_data)
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def check_step(self):
        if self.step == self.sample_num:
            self.step = 0
            random.shuffle(self.index_list)

    def sample(self, num):
        task = [self.task_id] * num
        inputs, outputs = [], []

        users = []
        for _ in range(num):
            self.check_step()
            idx = self.index_list[self.step]
            record = self.exp_data[idx]
            template = random.choice(exp_templates)
            inputs.append(template.format(record['user'], record['item']))
            outputs.append(record['explanation'])

            users.append(record['user'])
            self.step += 1
        return task, inputs, outputs, users


class SeqSampler:
    def __init__(self, user2items_pos):
        self.task_id = 1
        self.max_seq_len = 21
        self.item_template = ' item_'

        self.user2items_pos = user2items_pos
        self.user_list = list(user2items_pos.keys())

        self.sample_num = len(self.user_list)
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def check_step(self):
        if self.step == self.sample_num:
            self.step = 0
            random.shuffle(self.index_list)

    def sample_seq(self, u):
        item_history = self.user2items_pos[u]  # should have at least 4 items
        start_item = random.randint(0, len(item_history) - 4)  # cannot be the last 3
        end_item = random.randint(start_item + 1, len(item_history) - 3)  # cannot be the last 2
        item_seg = item_history[start_item:(end_item + 1)]  # sample a segment from the sequence without the last two
        if len(item_seg) > self.max_seq_len:
            item_seg = item_seg[-self.max_seq_len:]
        return item_seg

    def sample(self, num):
        task = [self.task_id] * num
        inputs, outputs = [], []
        users = []
        for _ in range(num):
            self.check_step()
            idx = self.index_list[self.step]
            u = self.user_list[idx]
            item_seg = self.sample_seq(u)
            template = random.choice(seq_templates)
            input_seq = template.format(u, self.item_template.join(item_seg[:-1]))
            inputs.append(input_seq)
            outputs.append(item_seg[-1])

            users.append(u)
            self.step += 1
        return task, inputs, outputs, users


class TopNSampler:
    def __init__(self, user2items_pos, negative_num, item_num):
        self.task_id = 2
        self.item_template = ' item_'
        self.negative_num = negative_num
        self.item_num = item_num

        self.user2item_set_pos = {}
        self.user2items_train = {}
        self.user_list = list(user2items_pos.keys())
        for user, items in user2items_pos.items():
            self.user2item_set_pos[user] = set([int(item) for item in items])
            self.user2items_train[user] = items[:-2]

        self.sample_num = len(self.user_list)
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def check_step(self):
        if self.step == self.sample_num:
            self.step = 0
            random.shuffle(self.index_list)

    def sample_negative(self, user):
        item_set = set()
        items_pos = self.user2item_set_pos[user]
        while len(item_set) < self.negative_num:
            i = random.randint(1, self.item_num)
            if i not in items_pos:
                item_set.add(i)
        return [str(item) for item in item_set]

    def sample(self, num):
        task = [self.task_id] * num
        inputs, outputs = [], []

        users = []
        for _ in range(num):
            self.check_step()
            idx = self.index_list[self.step]
            u = self.user_list[idx]
            item_list = self.user2items_train[u]
            item_pos = random.choice(item_list)
            item_list_neg = self.sample_negative(u)
            item_list_neg.append(item_pos)
            random.shuffle(item_list_neg)
            template = random.choice(topn_templates)
            input_seq = template.format(u, self.item_template.join(item_list_neg))
            inputs.append(input_seq)
            outputs.append(item_pos)

            users.append(u)
            self.step += 1
        return task, inputs, outputs, users

class TrainBatchify:
    def __init__(self, exp_data, user2items_pos, negative_num, item_num, tokenizer, exp_len, batch_size,data_dir):

        self.user_list = list(user2items_pos.keys())
        self.user_num = len(self.user_list)

        self.exp_sampler = ExpSampler(exp_data) 
        self.seq_sampler = SeqSampler(user2items_pos)
        self.topn_sampler = TopNSampler(user2items_pos, negative_num, item_num)
        self.tokenizer = tokenizer
        self.exp_len = exp_len
        self.batch_size = batch_size
        self.exp_num = int(self.exp_sampler.sample_num / batch_size)
        self.seq_num = int(self.seq_sampler.sample_num / batch_size)
        self.topn_num = int(self.topn_sampler.sample_num / batch_size)
        self.batch_num = self.exp_num + self.seq_num + self.topn_num
        self.batch_index = 0

        '''my'''
        file_path = data_dir+"user_timestamps_sorted.txt"  # 请替换为你的文件路径
        self.timestamps = self.read_timestamps_from_file(file_path)

    def encode(self, task, input_list, output_list, users):
        encoded_source = self.tokenizer(input_list, padding=True, return_tensors='pt')
        source_seq = encoded_source['input_ids'].contiguous()
        source_mask = encoded_source['attention_mask'].contiguous()
        max_len = source_seq.size(1)
        whole_word_ids, whole_word_ids_users = compute_whole_word_id(task[0], input_list, self.tokenizer, max_len, self.user_num)
        whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()
        if whole_word_ids_users is not None:
            whole_word_ids_users = torch.tensor(whole_word_ids_users, dtype=torch.int64).contiguous()
        encoded_target = self.tokenizer(output_list, padding=True, return_tensors='pt')
        target_seq = encoded_target['input_ids'][:, :self.exp_len]
        task = torch.tensor(task, dtype=torch.int64)
        return task, source_seq, source_mask, whole_word, whole_word_ids_users, target_seq, users
    
    def read_timestamps_from_file(self,file_path):
        timestamps = []
        
        with open(file_path, "r") as file:
            for line in file:
                # 按 `:` 拆分，获取用户 ID 和时间戳列表
                user_id, time_str = line.strip().split(": ")
                
                # 解析时间戳为整数列表
                time_list = list(map(int, time_str.split()))
                
                # 添加到 timestamps
                timestamps.append(time_list)
        
        return timestamps

    def next_batch(self):
        self.batch_index += 1
        if self.batch_index % 3 == 1:
            task_list, input_list, output_list, users = self.exp_sampler.sample(self.batch_size)
        elif self.batch_index % 3 == 2:
            task_list, input_list, output_list, users = self.seq_sampler.sample(self.batch_size)
        else:
            task_list, input_list, output_list, users = self.topn_sampler.sample(self.batch_size)
        return self.encode(task_list, input_list, output_list, users)


class ExpBatchify:
    def __init__(self, exp_data, user2items_pos, tokenizer, exp_len, batch_size):

        self.user_list = list(user2items_pos.keys())
        self.user_num = len(self.user_list)

        self.task_id = 0
        template = 'user_{} item_{}'
        input_list, output_list = [], []

        self.users = []
        for x in exp_data:
            input_list.append(template.format(x['user'], x['item']))
            self.users.append(x['user'])
            output_list.append(x['explanation'])

        encoded_source = tokenizer(input_list, padding=True, return_tensors='pt')
        self.source_seq = encoded_source['input_ids'].contiguous()
        self.source_mask = encoded_source['attention_mask'].contiguous()
        max_len = self.source_seq.size(1)
        whole_word_ids,_ = compute_whole_word_id(self.task_id, input_list, tokenizer, max_len, self.user_num)
        self.whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()
        encoded_target = tokenizer(output_list, padding=True, return_tensors='pt')
        self.target_seq = encoded_target['input_ids'][:, :exp_len].contiguous()
        self.batch_size = batch_size
        self.sample_num = len(exp_data)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        source_seq = self.source_seq[start:offset]  # (batch_size, seq_len)
        source_mask = self.source_mask[start:offset]
        whole_word = self.whole_word[start:offset]
        target_seq = self.target_seq[start:offset]
        task = torch.ones((offset - start,), dtype=torch.int64) * self.task_id

        user = self.users[start: offset]
        return task, source_seq, source_mask, whole_word,None, target_seq, user

    def next_batch_valid(self):
        return self.next_batch()

    def next_batch_test(self):
        return self.next_batch()


class SeqBatchify:
    def __init__(self, user2items_pos, tokenizer, batch_size, data_dir):
        self.task_id = 1
        self.max_seq_len = 21
        self.user_template = 'user_{} item_{}'
        self.item_template = ' item_'

        self.tokenizer = tokenizer
        self.user2items_pos = user2items_pos
        self.user_list = list(user2items_pos.keys())

        self.batch_size = batch_size
        self.sample_num = len(self.user_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

        self.user_list = list(user2items_pos.keys())
        self.user_num = len(self.user_list)

    def encode(self, input_list, output_list, users, valid = True):
        sample_num = len(input_list)
        encoded_source = self.tokenizer(input_list, padding=True, return_tensors='pt')
        source_seq = encoded_source['input_ids'].contiguous()
        source_mask = encoded_source['attention_mask'].contiguous()
        max_len = source_seq.size(1)

        whole_word_ids, whole_word_ids_users = compute_whole_word_id(self.task_id, input_list, self.tokenizer, max_len, self.user_num)
        whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()
        whole_word_users = torch.tensor(whole_word_ids_users, dtype=torch.int64).contiguous()
        encoded_target = self.tokenizer(output_list, padding=True, return_tensors='pt')
        target_seq = encoded_target['input_ids']
        task = torch.ones((sample_num,), dtype=torch.int64) * self.task_id
        if valid:
            return task, source_seq, source_mask, whole_word, whole_word_users, target_seq, users
        else:
            return task, source_seq, source_mask, whole_word, whole_word_users, target_seq

    def next_batch(self, valid=True):
        if self.step == self.total_step:
            self.step = 0

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1

        input_list = []
        output_list = []
        users = []
        for i in range(start, offset):
            u = self.user_list[i]
            item_seg = self.user2items_pos[u]
            if valid:
                item_seg = item_seg[:-1]  # leave the last 1
            if len(item_seg) > self.max_seq_len:
                item_seg = item_seg[-self.max_seq_len:]
            input_seq = self.user_template.format(u, self.item_template.join(item_seg[:-1]))

            input_list.append(input_seq)
            output_list.append(item_seg[-1])

            users.append(u)

        return self.encode(input_list, output_list, users, valid)

    def next_batch_valid(self):
        return self.next_batch()

    def next_batch_test(self):
        return self.next_batch(False)


class TopNBatchify:
    def __init__(self, user2items_pos, user2items_neg, negative_num, item_num, tokenizer, batch_size=128):
        self.task_id = 2
        self.user_template = 'user_{} item_{}'
        self.item_template = ' item_'
        self.negative_num = negative_num
        self.item_num = item_num

        self.tokenizer = tokenizer
        self.user2items_neg = user2items_neg
        self.user2item_set_pos = {}
        self.user2item_val = {}
        self.user2item_test = {}
        self.user_list = list(user2items_pos.keys())
        self.user_num = len(self.user_list)

        for user, items in user2items_pos.items():
            self.user2item_set_pos[user] = set([int(item) for item in items])
            self.user2item_val[user] = items[-2]
            self.user2item_test[user] = items[-1]

        self.batch_size = batch_size
        self.sample_num = len(self.user_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def encode(self, input_list, output_list, users, valid=True):
        sample_num = len(input_list)
        encoded_source = self.tokenizer(input_list, padding=True, return_tensors='pt')
        source_seq = encoded_source['input_ids'].contiguous()
        source_mask = encoded_source['attention_mask'].contiguous()
        max_len = source_seq.size(1)
        whole_word_ids,_ = compute_whole_word_id(self.task_id, input_list, self.tokenizer, max_len, self.user_num)
        whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()
        encoded_target = self.tokenizer(output_list, padding=True, return_tensors='pt')
        target_seq = encoded_target['input_ids']
        task = torch.ones((sample_num,), dtype=torch.int64) * self.task_id
        if valid:
            return task, source_seq, source_mask, whole_word,None, target_seq, users
        else:
            return task, source_seq, source_mask, whole_word,None, target_seq
    def sample_negative(self, user):
        item_set = set()
        items_pos = self.user2item_set_pos[user]
        while len(item_set) < self.negative_num:
            i = random.randint(1, self.item_num)
            if i not in items_pos:
                item_set.add(i)
        return [str(item) for item in item_set]

    def next_batch(self, valid=True):
        if self.step == self.total_step:
            self.step = 0

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1

        input_list = []
        output_list = []
        users = []
        for i in range(start, offset):
            u = self.user_list[i]
            if valid:
                item_pos = self.user2item_val[u]
                item_list_neg = self.sample_negative(u)
            else:
                item_pos = self.user2item_test[u]
                item_list_neg = self.user2items_neg[u]
            item_list_neg.append(item_pos)
            random.shuffle(item_list_neg)
            input_seq = self.user_template.format(u, self.item_template.join(item_list_neg))
            input_list.append(input_seq)
            output_list.append(item_pos)

            users.append(u)
        return self.encode(input_list, output_list, users, valid)

    def next_batch_valid(self):
        return self.next_batch()

    def next_batch_test(self):
        return self.next_batch(False)


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def evaluate_ndcg(user2item_test, user2items_top, top_k):
    dcgs = [1 / math.log2(i + 2) for i in range(top_k)]
    ndcg = 0
    for u, items in user2items_top.items():
        ground_truth = set(user2item_test[u])
        dcg = 0
        count = 0
        for idx, item in enumerate(items[:top_k]):
            if item in ground_truth:
                dcg += dcgs[idx]
                count += 1
        if count > 0:
            dcg = dcg / sum(dcgs[:count])
        ndcg += dcg
    return ndcg / len(user2item_test)


def evaluate_hr(user2item_test, user2items_top, top_k):
    total = 0
    for u, items in user2items_top.items():
        ground_truth = set(user2item_test[u])
        count = 0
        for item in items[:top_k]:
            if item in ground_truth:
                count += 1
        total += count / len(ground_truth)

    return total / len(user2item_test)
