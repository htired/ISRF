import math
import json
import numpy as np
import torch
import random
import datetime
from util1.rouge import rouge
from util1.bleu import compute_bleu
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

        nitem = len(self.id2item)
        self.pop = {}  # 初始化流行度统计的字典
        with open(data_dir + 'sequential.txt', 'r') as f:
            for line in f.readlines():
                user, items = line.strip().split(' ', 1)  # 分割用户和物品列表
                items = items.split(' ')  # 将物品列表分割成单个物品
                for item in items:  # 遍历每个物品
                    item = int(item)  # 将物品字符串转换为整数
                    if item not in self.pop:
                        self.pop[item] = 0  # 初始化该物品的计数
                    self.pop[item] += 1  # 更新该物品的计数

def compute_whole_word_id(task_id, seq_batch, tokenizer, max_len, user_num, alpha=1.0, decay_type="exp"):
    whole_word_ids = []
    whole_word_ids_users = []
    for seq_idx, seq in enumerate(seq_batch):
        token_list = tokenizer.tokenize(seq)
        start_indices = []
        
        # 1️⃣ 识别 "user_xx" 或 "item_xx" 起始位置
        for idx, token in enumerate(token_list):
            if token == '_':
                start_indices.append(idx - 1)  # user_xx 或 item_xx 的起始索引
        
        # 2️⃣ 计算 ID 范围
        end_indices = []
        for start in start_indices:
            mover = start + 2  # user/item _ xx
            while mover < len(token_list) and token_list[mover].isdigit():
                mover += 1
            end_indices.append(mover)

        whole_word_id = [0] * len(token_list)  # 初始化全 0，作为 padding
        whole_word_id_user = [0] * len(token_list)  # 初始化全 0，作为 padding
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
    # 5️⃣ 进行 Padding
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


# class TrainBatchify:
#     def __init__(self, exp_data, user2items_pos, negative_num, item_num, tokenizer, exp_len, batch_size):

#         self.user_list = list(user2items_pos.keys())
#         self.user_num = len(self.user_list)

#         self.exp_sampler = ExpSampler(exp_data) 
#         self.seq_sampler = SeqSampler(user2items_pos)
#         self.topn_sampler = TopNSampler(user2items_pos, negative_num, item_num)
#         self.tokenizer = tokenizer
#         self.exp_len = exp_len
#         self.batch_size = batch_size
#         self.exp_num = int(self.exp_sampler.sample_num / batch_size)
#         self.seq_num = int(self.seq_sampler.sample_num / batch_size)
#         self.topn_num = int(self.topn_sampler.sample_num / batch_size)
#         self.batch_num = self.exp_num + self.seq_num + self.topn_num
#         self.batch_index = 0

#     def encode(self, task, input_list, output_list, users):
#         encoded_source = self.tokenizer(input_list, padding=True, return_tensors='pt')
#         source_seq = encoded_source['input_ids'].contiguous()
#         source_mask = encoded_source['attention_mask'].contiguous()
#         max_len = source_seq.size(1)
#         whole_word_ids = compute_whole_word_id(task[0], input_list, self.tokenizer, max_len, self.user_num)
#         whole_word = torch.tensor(whole_word_ids, dtype=torch.int64).contiguous()
#         encoded_target = self.tokenizer(output_list, padding=True, return_tensors='pt')
#         target_seq = encoded_target['input_ids'][:, :self.exp_len]
#         task = torch.tensor(task, dtype=torch.int64)
#         return task, source_seq, source_mask, whole_word, target_seq, users


#     def next_batch(self):
#         self.batch_index += 1
#         if self.batch_index % 3 == 1:
#             task_list, input_list, output_list, users = self.exp_sampler.sample(self.batch_size)
#         elif self.batch_index % 3 == 2:
#             task_list, input_list, output_list, users = self.seq_sampler.sample(self.batch_size)
#         else:
#             task_list, input_list, output_list, users = self.topn_sampler.sample(self.batch_size)
#         return self.encode(task_list, input_list, output_list, users)


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
        '''my'''
        file_path = data_dir+"user_timestamps_sorted.txt"  # 请替换为你的文件路径
        self.timestamps = self.read_timestamps_from_file(file_path)
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


# user2item_test: {1：[], 2: []}
# user2item_test: {1：[], 2: []}
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


# user2item_test: {1：[], 2: []}
# user2item_test: {1：[], 2: []}
def evaluate_hr_long_short(user2item_test, user2items_top, user2item_len, top_k, ts_user=10, aug_len=0):
    """
    长短尾用户的 Hit Rate 评估

    参数：
    - user2item_test: dict，用户对应的测试集物品列表
    - user2items_top: dict，用户对应的推荐物品列表
    - user2item_len: dict，用户的行为数据长度
    - top_k: int，Top-K 推荐中 K 的值
    - ts_user: int，分割长尾和短尾用户的阈值
    - aug_len: int，附加的长度修正，用于调节长短尾分割
    返回：
    - dict：包含长尾和短尾用户的 HR@K 指标
    """
    # 初始化长尾和短尾用户的 HR 和计数
    HR_short, HR_long = 0, 0
    count_short, count_long = 0, 0

    for u, items in user2items_top.items():
        # Ground truth：测试集中的真实物品
        ground_truth = set(user2item_test[u])
        if len(ground_truth) == 0:
            continue  # 如果 ground_truth 为空，则跳过此用户

        # 检查用户行为数据长度，判断是长尾还是短尾
        user_len = user2item_len[u]
        if user_len < ts_user + aug_len:
            # 短尾用户
            count_short += 1
            count = 0
            for item in items[:top_k]:
                if item in ground_truth:
                    count += 1
            HR_short += count / len(ground_truth)
        else:
            # 长尾用户
            count_long += 1
            count = 0
            for item in items[:top_k]:
                if item in ground_truth:
                    count += 1
            HR_long += count / len(ground_truth)

    # 避免除以 0
    HR_short = HR_short / count_short if count_short != 0 else 0
    HR_long = HR_long / count_long if count_long != 0 else 0

    return {
        "Short HR@{}".format(top_k): HR_short,
        "Long HR@{}".format(top_k): HR_long,
    }




def evaluate_ndcg_long_short(user2item_test, user2items_top, user2item_len, top_k, ts_user=10, aug_len=0):
    """
    计算长尾和短尾用户的 NDCG 指标

    参数：
    - user2item_test: dict，用户对应的测试集物品列表
    - user2items_top: dict，用户对应的推荐物品列表
    - user2item_len: dict，用户的行为数据长度
    - top_k: int，Top-K 推荐中 K 的值
    - ts_user: int，分割长尾和短尾用户的阈值
    - aug_len: int，附加的长度修正，用于调节长短尾分割
    返回：
    - dict：包含长尾和短尾用户的 NDCG@K 指标
    """
    # 预计算每个排名位置的折扣因子
    dcgs = [1 / math.log2(i + 2) for i in range(top_k)]

    # 初始化长尾和短尾用户的 NDCG 和计数
    NDCG_short, NDCG_long = 0, 0
    count_short, count_long = 0, 0

    for u, items in user2items_top.items():
        # Ground truth：测试集中的真实物品
        ground_truth = set(user2item_test[u])
        if len(ground_truth) == 0:
            continue  # 如果 ground_truth 为空，则跳过此用户

        # 检查用户行为数据长度，判断是长尾还是短尾
        user_len = user2item_len[u]
        dcg = 0
        count = 0
        for idx, item in enumerate(items[:top_k]):
            if item in ground_truth:
                dcg += dcgs[idx]
                count += 1

        if count > 0:
            dcg = dcg / sum(dcgs[:count])  # 归一化 DCG

        if user_len < ts_user + aug_len:
            # 短尾用户
            NDCG_short += dcg
            count_short += 1
        else:
            # 长尾用户
            NDCG_long += dcg
            count_long += 1

    # 避免除以 0
    NDCG_short = NDCG_short / count_short if count_short != 0 else 0
    NDCG_long = NDCG_long / count_long if count_long != 0 else 0

    return {
        "Short NDCG@{}".format(top_k): NDCG_short,
        "Long NDCG@{}".format(top_k): NDCG_long,
    }


import torch
import math

def metric_pop_report(user2item_test, user2items_top, pop_dict, target_items, topk=10, aug_pop=0, args=None):
    """
    Report the metrics according to target item's popularity.

    Parameters:
    - user2item_test: dict, 用户测试集中用户-物品对应关系
    - user2items_top: dict, 用户推荐列表
    - pop_dict: dict, mapping of item IDs to their popularity scores
    - target_items: Tensor, the target items to evaluate (must be a Tensor)
    - topk: int, the Top-K ranking threshold (default is 10)
    - aug_pop: int, adjustment to popularity threshold (default is 0)
    - args: optional, contains additional parameters such as ts_item

    Returns:
    - dict: Contains NDCG and HR metrics for tail and popular items.
    """
    if args is not None:
        ts_tail = args.ts_item
    else:
        ts_tail = 20

    # Ensure target_items is a list or tensor of integers
    target_items = target_items.to(torch.int64).tolist()

    # Compute item popularity from pop_dict
    item_pop = torch.tensor([pop_dict.get(item, 0) for item in target_items], dtype=torch.float32)

    # Determine counts for tail and popular items
    count_s = len(item_pop[item_pop < ts_tail + aug_pop])
    count_l = len(item_pop[item_pop >= ts_tail + aug_pop])

    # Initialize metrics
    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0

    # Iterate over user recommendations and compute NDCG and HR
    for user, top_items in user2items_top.items():
        ground_truth = set(user2item_test.get(user, []))  # 用户真实的目标物品集
        if len(ground_truth) == 0:
            continue  # 如果用户没有目标物品，则跳过

        for rank, item in enumerate(top_items[:topk]):
            if item in ground_truth:
                # Check item's popularity
                item_popularity = pop_dict.get(item, 0)
                if item_popularity < ts_tail + aug_pop:
                    # Tail items
                    NDCG_s += 1 / math.log2(rank + 2)
                    HT_s += 1
                else:
                    # Popular items
                    NDCG_l += 1 / math.log2(rank + 2)
                    HT_l += 1

    # Avoid division by zero and compute metrics
    return {
        f'Tail NDCG@{topk}': NDCG_s / count_s if count_s != 0 else 0,
        f'Tail HR@{topk}': HT_s / count_s if count_s != 0 else 0,
        f'Popular NDCG@{topk}': NDCG_l / count_l if count_l != 0 else 0,
        f'Popular HR@{topk}': HT_l / count_l if count_l != 0 else 0,
    }


def ids2tokens(ids, tokenizer):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.split()

import math

def evaluate_long_short_metrics(user2item_test, user2items_top, user2item_len, top_k, ts_user=10, aug_len=0):
    """
    计算长尾和短尾用户的 HR 和 NDCG 指标

    参数：
    - user2item_test: dict，用户对应的测试集物品列表
    - user2items_top: dict，用户对应的推荐物品列表
    - user2item_len: dict，用户的行为数据长度
    - top_k: int，Top-K 推荐中 K 的值
    - ts_user: int，分割长尾和短尾用户的阈值
    - aug_len: int，附加的长度修正，用于调节长短尾分割
    返回：
    - dict：包含长尾和短尾用户的 HR@K 和 NDCG@K 指标
    """
    # 预计算每个排名位置的折扣因子
    dcgs = [1 / math.log2(i + 2) for i in range(top_k)]

    # 初始化长尾和短尾用户的 HR 和 NDCG 以及计数
    HR_short, HR_long = 0, 0
    NDCG_short, NDCG_long = 0, 0
    count_short, count_long = 0, 0

    for u, items in user2items_top.items():
        # Ground truth：测试集中的真实物品
        ground_truth = set(user2item_test[u])
        if len(ground_truth) == 0:
            continue  # 如果 ground_truth 为空，则跳过此用户

        # 检查用户行为数据长度，判断是长尾还是短尾
        user_len = user2item_len[u]
        dcg = 0
        hr_count = 0

        for idx, item in enumerate(items[:top_k]):
            if item in ground_truth:
                dcg += dcgs[idx]
                hr_count += 1

        if hr_count > 0:
            dcg = dcg / sum(dcgs[:hr_count])  # 归一化 DCG

        if user_len < ts_user + aug_len:
            # 短尾用户
            HR_short += hr_count / len(ground_truth)
            NDCG_short += dcg
            count_short += 1
        else:
            # 长尾用户
            HR_long += hr_count / len(ground_truth)
            NDCG_long += dcg
            count_long += 1

    # 避免除以 0
    HR_short = HR_short / count_short if count_short != 0 else 0
    HR_long = HR_long / count_long if count_long != 0 else 0
    NDCG_short = NDCG_short / count_short if count_short != 0 else 0
    NDCG_long = NDCG_long / count_long if count_long != 0 else 0

    return {
        "Short HR@{}".format(top_k): HR_short,
        "Long HR@{}".format(top_k): HR_long,
        "Short NDCG@{}".format(top_k): NDCG_short,
        "Long NDCG@{}".format(top_k): NDCG_long,
    }


import math

def evaluate_long_short_metrics5(user2item_test, 
                                 user2items_top, 
                                 user2item_len, 
                                 top_k, 
                                 thresholds=[5, 10, 15, 20],
                                   aug_len=0):
    """
    计算长尾和短尾用户的 HR 和 NDCG 指标，按多个阈值对用户进行分组。

    参数：
    - user2item_test: dict，用户对应的测试集物品列表
    - user2items_top: dict，用户对应的推荐物品列表
    - user2item_len: dict，用户的行为数据长度
    - top_k: int，Top-K 推荐中 K 的值
    - thresholds: list[int]，分割长尾和短尾用户的阈值
    - aug_len: int，附加的长度修正，用于调节长短尾分割
    返回：
    - dict：包含各个组别的 HR@K 和 NDCG@K 指标
    """
    # 预计算每个排名位置的折扣因子
    dcgs = [1 / math.log2(i + 2) for i in range(top_k)]

    # 初始化指标
    NDCG = np.zeros(5)
    HR = np.zeros(5)
    count = np.zeros(5)
    max_threshold = int(1e6)  # A large integer to cover all users with length > the last threshold
    thresholds = thresholds + [max_threshold]  # Append the max_threshold to the end of the list

    for u, items in user2items_top.items():
        # Ground truth：测试集中的真实物品
        ground_truth = set(user2item_test[u])
        if len(ground_truth) == 0:
            continue  # 如果 ground_truth 为空，则跳过此用户

        # 用户的行为数据长度
        user_len = user2item_len[u]
        
        # 计算 DCG 和 HR
        dcg = 0
        hr_count = 0
        for idx, item in enumerate(items[:top_k]):
            if item in ground_truth:
                dcg += dcgs[idx]
                hr_count += 1

        if hr_count > 0:
            dcg = dcg / sum(dcgs[:hr_count])  # 归一化 DCG

        # 根据行为数据长度，判断用户属于哪个组
        for i in range(5):
            if user_len < thresholds[i] + aug_len:
                NDCG[i] += dcg
                HR[i] += hr_count / len(ground_truth)
                count[i] += 1
                break

    # 避免除以 0
    for i in range(5):
        if count[i] != 0:
            NDCG[i] /= count[i]
            HR[i] /= count[i]
        else:
            NDCG[i] = 0
            HR[i] = 0

    # 返回每个组别的指标
    result = {}
    for i in range(5):
        result[f"Group {i+1} HR@{top_k}"] = HR[i]
        result[f"Group {i+1} NDCG@{top_k}"] = NDCG[i]

    return result

def metric_pop_5group(user2item_test, 
                      user2items_top, 
                      pop_dict, 
                      target_items, 
                      topk=10, 
                      thresholds = [10, 20, 30, 40]):
    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    # Ensure target_items is a list or tensor of integers
    target_items = target_items.to(torch.int64).tolist()

    # Iterate over user recommendations and compute NDCG and HR
    for user, top_items in user2items_top.items():
        ground_truth = set(user2item_test.get(user, []))  # 用户真实的目标物品集
        if len(ground_truth) == 0:
            continue  # 如果用户没有目标物品，则跳过

        for rank, item in enumerate(top_items[:topk]):
            if item in ground_truth:
                # Check item's popularity
                item_popularity = pop_dict.get(item, 0)
                if item_popularity < thresholds[0]:
                    NDCG[0] += 1 / np.log2(rank + 2)
                    HR[0] += 1

                elif item_popularity < thresholds[1]:
                    NDCG[1] += 1 / np.log2(rank + 2)
                    HR[1] += 1

                elif item_popularity < thresholds[2]:
                    NDCG[2] += 1 / np.log2(rank + 2)
                    HR[2] += 1

                elif item_popularity < thresholds[3]:
                    NDCG[3] += 1 / np.log2(rank + 2)
                    HR[3] += 1

                else:
                    NDCG[4] += 1 / np.log2(rank + 2)
                    HR[4] += 1
    # Compute item popularity from pop_dict
    pop = torch.tensor([pop_dict.get(item, 0) for item in target_items], dtype=torch.float32)
    count = np.zeros(5)

    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count
