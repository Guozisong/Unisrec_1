import argparse
import csv
import datetime
import json
import os
from tqdm import tqdm
from preprocessing_utils import filter_inters, make_inters_in_order, \
    generate_training_data, generate_item_embedding, convert_to_atomic_files
from preprocessing_runtime import check_path, set_device, load_plm


def load_ratings(file):
    inters = set()
    with open(file, 'r', encoding='utf-8') as fp:
        fp.readline()
        cr = csv.reader(fp)
        for line in tqdm(cr, desc='Load ratings'):
            try:
                user_id, item_id, event_value, event_time, item_text = line
                if '' in (user_id, item_id, event_value, event_time, item_text):
                    continue
                ts = datetime.datetime.strptime(event_time, '%Y-%m-%d').timestamp()
                inters.add((user_id, item_id, float(event_value), int(ts)))
            except ValueError:
                print(line)
    return inters


def preprocess_rating(args):
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    rating_file_path = os.path.join(args.input_path, f'{args.dataset}.csv')
    rating_inters = load_ratings(rating_file_path)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, user_k_core_threshold_min=args.user_k_min,
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
        for line in tqdm(cr, desc='Load item text'):
            try:
                user_id, item_id, event_value, event_time, item_text = line
                if '' in (user_id, item_id, event_value, event_time, item_text):
                    continue
                if item_id not in item2text:
                    item2text[item_id] = item_text
            except ValueError:
                print(line)

    for iid in tqdm(items, desc='Generate text'):
        assert iid in item2text
        text = item2text[iid].strip().lower() + '.'
        item_text_list.append([iid, text])
    return item_text_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--user_k_min', type=int, default=3, help='user k-core filtering')
    parser.add_argument('--user_k_max', type=int, default=500, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--max_seq_length', type=int, default=50, help='recent interactions retained per user (3-100)')
    parser.add_argument('--input_path', type=str, default='outputs/raw/')
    parser.add_argument('--output_path', type=str, default='outputs/downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='./bert-base-uncased/')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')

    args = parser.parse_args()
    if not 3 <= args.max_seq_length <= 100:
        parser.error('--max_seq_length must be from 3 to 100')
    return args

def main(args):
    rating_inters = preprocess_rating(args)
    item_text_list = generate_text(args, {item for _, item, _, _ in rating_inters})
    train_inters, valid_inters, test_inters, predict_inters, user2index, item2index = \
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

    # 训练、验证、测试和生产预测序列构建并保存
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters, predict_inters)


if __name__ == '__main__':
    args = parse_args()
    main(args)
