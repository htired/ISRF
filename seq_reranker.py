import json
import os

import numpy as np
import torch
import random
import argparse
from transformers import T5Tokenizer
from util1.utils import SeqDataLoader, SeqBatchify, now_time, evaluate_ndcg, evaluate_hr,evaluate_hr_long_short,evaluate_ndcg_long_short,metric_pop_report


parser = argparse.ArgumentParser(description='ELMRec')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./ELMRec/',
                    help='directory to load the final model')
parser.add_argument('--num_beams', type=int, default=30,
                    help='number of beams')
parser.add_argument('--top_n', type=int, default=10,
                    help='number of items to predict')

# both tasks : {'Sports': 10, 'Beauty': 15, 'Toys': 10}
parser.add_argument('--N', type=int, default=10,
                    help='number of additional candidates')

parser.add_argument('--knn_a', type=int, default=6,
                    help='')


parser.add_argument('--model_saved_name', type=str, default='model',
                    help='')
parser.add_argument('--data_name', type=str, default='beauty',
                    help='beauty, sports, toys')

parser.add_argument('--gpu', type=int, default='0',
                    help='0; 1')
parser.add_argument('--seed', type=int, default=10,
                    help='number of items to predict')
args = parser.parse_args()

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)

seed_it(args.seed)

if args.model_version == 1:
    model_version = 't5-base'
elif args.model_version == 2:
    model_version = 't5-large'
elif args.model_version == 3:
    model_version = 't5-3b'
elif args.model_version == 4:
    model_version = 't5-11b'
else:
    model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda:{}'.format(args.gpu) if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, '{}.pt'.format(args.model_saved_name))

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
tokenizer = T5Tokenizer.from_pretrained(model_version)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)

###############################################################################
# Test the model
###############################################################################

def generate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, _ = seq_iterator.next_batch_test()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)

            beam_outputs = model.beam_search(task, source, whole_word, source_mask,
                                             num_beams=args.num_beams,
                                             num_return_sequences=args.top_n + args.N,
                                             )

            output_tensor = beam_outputs.view(task.size(0), args.top_n + args.N, -1)
            for i in range(task.size(0)):
                results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
                idss_predict.append(results)

            if seq_iterator.step == seq_iterator.total_step:
                break
    return idss_predict


# Load the best saved model.
with open(model_path, 'rb') as f:
    # model = torch.load(f).to(device)
    # 加载模型并将其放到 GPU 1
    model = torch.load(f, map_location=device)
    # import numpy as np
    # np.save("whole_word_embeddings_beauty_3_order.npy", model.graph_convolution_embeddings(3).detach().cpu().numpy())

# Run on test data.
print(now_time() + 'Generating recommendations')
user2item_test = {}
interacted_items = {}
user2item_len = {}

target_items = torch.empty(0)

for user, item_list in seq_corpus.user2items_positive.items():
    user2item_test[user] = [int(item_list[-1])]
    interacted_items[user] = [int(item_id) for item_id in item_list[:-1]]
    # 交互序列长度
    user2item_len[user] = len(item_list)
    target_items = torch.cat([target_items, torch.tensor([int(item_list[-1])])])
idss_predicted = generate()
print(now_time() + 'Evaluation')

user2rank_list = {}
for predictions, user in zip(idss_predicted, seq_iterator.user_list):

    prediction_list = []
    for p in predictions:
        try:
            predicted_item_id = int(p.split(' ')[0])  # use the id before white space
            if predicted_item_id not in interacted_items[user]:
                prediction_list.append(predicted_item_id)
        except:
            pass
    while len(prediction_list) < args.top_n:
        prediction_list.append(random.randint(1, nitem)) # randomly generate a recommendation

    user2rank_list[user] = prediction_list
output_dir = args.data_dir
# with open(os.path.join(output_dir, 'user2item_test_o.json'), 'w') as f:
#     json.dump(user2item_test, f)

# with open(os.path.join(output_dir, 'user2rank_list_o.json'), 'w') as f:
#     json.dump(user2rank_list, f)

# user2item_test = {
#     1: [10, 20],
#     2: [30],
#     3: [40, 50]
# }

# # 用户的推荐列表
# user2rank_list = {
#     1: [20, 30, 10],
#     2: [30, 40],
#     3: [50, 10]
# }
#
# # 物品的流行度
# pop_dict = {
#     10: 5,
#     20: 15,
#     30: 25,
#     40: 8,
#     50: 18
# }
#
# # 目标物品列表
# target_items = torch.tensor([10, 20, 30, 40, 50])
#
# # Top-K 阈值
# topk = 10
#
top_ns = [1]
if args.top_n >= 5:
    for i in range(1, (args.top_n // 5) + 1):
        top_ns.append(i * 5)

# Ensure the directory exists
os.makedirs(args.data_dir, exist_ok=True)

# File path for the result.txt file
result_file = os.path.join(args.data_dir, 'result.txt')

item_pop = seq_corpus.pop
# # 物品的流行度
# item_pop = {
#     10: 5,
#     20: 15,
#     30: 25,
#     40: 8,
#     50: 18
# }

with open(result_file, 'a') as f:
    f.write("{}\n".format(args.data_name))
    f.write("seq\n model: {}, checkpoint: {}\n".format(args.model_saved_name, args.checkpoint))

    for top_n in top_ns:
        # Overall Performance
        hr = evaluate_hr(user2item_test, user2rank_list, top_n)
        ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)

        # User Group Performance
        hr_ls = evaluate_hr_long_short(user2item_test, user2rank_list, user2item_len, top_n)
        ndcg_ls = evaluate_ndcg_long_short(user2item_test, user2rank_list, user2item_len, top_n)

        # Item Group Performance
        pop = metric_pop_report(user2item_test, user2rank_list, item_pop, target_items, top_n)

        result = (
            f"Overall Performance:\n"
            f"    NDCG@{top_n}: {ndcg:.5f}\n"
            f"    HR@{top_n}: {hr:.5f}\n\n"
            f"User Group Performance:\n"
            f"    Short NDCG@{top_n}: {ndcg_ls['Short NDCG@' + str(top_n)]:.5f}\n"
            f"    Short HR@{top_n}: {hr_ls['Short HR@' + str(top_n)]:.5f}\n"
            f"    Long NDCG@{top_n}: {ndcg_ls['Long NDCG@' + str(top_n)]:.5f}\n"
            f"    Long HR@{top_n}: {hr_ls['Long HR@' + str(top_n)]:.5f}\n\n"
            f"Item Group Performance:\n"
            f"    Tail NDCG@{top_n}: {pop['Tail NDCG@{}'.format(top_n)]:.5f}\n"
            f"    Tail HR@{top_n}: {pop['Tail HR@{}'.format(top_n)]:.5f}\n"
            f"    Popular NDCG@{top_n}: {pop['Popular NDCG@{}'.format(top_n)]:.5f}\n"
            f"    Popular HR@{top_n}: {pop['Popular HR@{}'.format(top_n)]:.5f}\n"
        )

        f.write(result)  # 写入文件
        print(result)  # 控制台打印
