import argparse
import csv
import datetime
import os
from tqdm import tqdm
from utils1_production_env import filter_inters, make_inters_in_order, get_user_item_from_ratings, \
    generate_training_data, generate_item_embedding, convert_to_atomic_files
from utils2 import check_path, set_device, load_plm
import time
import json


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r', encoding='utf-8') as fp:
        fp.readline()
        cr = csv.reader(fp)
        for line in tqdm(cr, desc='Load ratings'):
            try:
                # 用户id, 商品id, 购买数量, 购买时间, 商品属性
                user_id, prod_id, purchase_count, dt, attrvalues = line
                if '' in (user_id, prod_id, purchase_count, dt, attrvalues):
                    continue
                users.add(user_id)
                items.add(prod_id)
                ts = datetime.datetime.strptime(dt, '%Y-%m-%d').timestamp()
                inters.add((user_id, prod_id, float(purchase_count), int(ts)))
            except ValueError:
                print(line)
    return users, items, inters


def preprocess_rating(args):
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, f'{args.dataset}.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=rating_items,
                                  user_k_core_threshold_min=args.user_k_min,
                                  user_k_core_threshold_max=args.user_k_max,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    return rating_inters


def generate_text(args, items):
    item_text_list = []

    meta_file_path = os.path.join(args.input_path, f'{args.dataset}.csv')
    item2text = {}
    with open(meta_file_path, 'r', encoding='utf-8') as fp:
        fp.readline()
        cr = csv.reader(fp)
        for line in tqdm(cr, desc='Load ratings'):
            try:
                user_id, prod_id, purchase_count, dt, attrvalues = line
                if '' in (user_id, prod_id, purchase_count, dt, attrvalues):
                    continue
                if prod_id not in item2text:
                    item2text[prod_id] = attrvalues
            except ValueError:
                print(line)

    for iid in tqdm(items, desc='Generate text'):
        assert iid in item2text
        text = item2text[iid].strip().lower() + '.'
        item_text_list.append([iid, text])
    return item_text_list


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)

    # load item text and clean
    item_text_list = generate_text(args, rating_items)
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list

# user_k_max设置一个较大值，目的是不过滤购买序列较长的用户，实际上只截取所有序列后50个商品序列用于构造训练和测试集
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lianhua')
    parser.add_argument('--user_k_min', type=int, default=3, help='user k-core filtering')
    parser.add_argument('--user_k_max', type=int, default=500, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='/ml/output/raw/')
    parser.add_argument('--output_path', type=str, default='/ml/output/downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='./bert-base-uncased/')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')

    parser.add_argument("--input1", type=str, default=None, help="Component input port 1.")
    parser.add_argument("--input2", type=str, default=None, help="Component input port 2.")
    parser.add_argument("--input3", type=str, default=None, help="Component input port 3.")
    parser.add_argument("--input4", type=str, default=None, help="Component input port 4.")
    parser.add_argument("--output1", type=str, default=None, help="Output OSS port 1.")
    parser.add_argument("--output2", type=str, default=None, help="Output OSS port 2.")
    parser.add_argument("--output3", type=str, default=None, help="Output MaxComputeTable 1.")
    parser.add_argument("--output4", type=str, default=None, help="Output MaxComputeTable 2.")

    return parser.parse_args()

def main(args):
    s = time.time()
    # 初始参数
    """
    preprocess_rating
        (一) 对用户-商品交互(inners)进行过滤 1、用户产生的inner满足阈值 2、商品相关的inner满足阈值  
        (二) 使得用户-商品在交互时间上有序，
    rating_inters 用户-商品时序交互列表  
        [('963efdd263be4a8a97b3b7948a9a2074', '221306', 1.0, 1743073892), 
         ('963efdd263be4a8a97b3b7948a9a2074', '86406', 2.0, 1743073892),
         ('963efdd263be4a8a97b3b7948a9a2074', '93706', 1.0, 1743678338),
         ...]
    """
    rating_inters = preprocess_rating(args)

    # 商品id-商品属性列表 [['82633', '丹夫（danco） 巧克力味 丹夫巧克力薄脆 88g/盒.'], ...]
    item_text_list = preprocess_text(args, rating_inters)
    """
    generate_training_data
        (一) 对用户、商品进行编号, user2index {用户id: 编号}, item2index {商品id: 编号}
        (二) 对于每一个用户, 时序上最后一个用户-商品交互用于测试, 倒数第二个交互用于验证, 之前的所有交互用于训练
        (三) train_inter/valid_inters/test_inters {用户编号: [交互商品编号]}
    """
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters)

    index2user = {v: k for k, v in user2index.items()}
    index2item = {v: k for k, v in item2index.items()}


    # GPU/CPU设备 & 预训练编码模型初始化
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # 创建输出存储目录
    check_path(os.path.join(args.output_path, args.dataset))

    # 保存index2user、index2item
    with open(os.path.join(args.output_path, args.dataset, "index2user.json"), 'w', encoding='utf-8') as f:
        json.dump(index2user, f, ensure_ascii=False, indent=4)
    with open(os.path.join(args.output_path, args.dataset, "index2item.json"), 'w', encoding='utf-8') as f:
        json.dump(index2item, f, ensure_ascii=False, indent=4)

    # 使用模型对所有商品描述文本进行embedding编码(编码向量行索引为商品编码)并保存  Embeddings shape: (商品数量, 编码维度)
    generate_item_embedding(args, item_text_list, item2index,
                            plm_tokenizer, plm_model)
    if args.word_drop_ratio > 0:
        generate_item_embedding(args, item_text_list, item2index,
                                plm_tokenizer, plm_model, word_drop_ratio=args.word_drop_ratio)

    # 训练、验证、测试集构建并保存
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    e = time.time()

    # print('编码用时：{} min'.format((e - s) / 60))


if __name__ == '__main__':
    args = parse_args()
    main(args)
