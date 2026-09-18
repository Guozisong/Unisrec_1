import collections
import os
import random
import torch


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold_min, threshold_max=100000000):
    cans = set()
    for unit, count in unit2count.items():
        if threshold_min <= count <= threshold_max:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, user_k_core_threshold_min=0,
                  user_k_core_threshold_max=0, item_k_core_threshold=0):
    new_inters = []
    # filter by k-core
    if user_k_core_threshold_min or user_k_core_threshold_max or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold_min, user_k_core_threshold_max)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                  % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset, '\n')

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters, predict_inters = dict(), dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index][-args.max_seq_length:]
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        predict_inters[u_index] = [str(i_index) for i_index in inters]
    return train_inters, valid_inters, test_inters, predict_inters, user2index, item2index


def generate_item_embedding(args, item_text_list, item2index, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            # print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:, 1:, :].sum(dim=1) / \
                          encoded_sentences['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape, '\n')

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


# {用户编号: [交互商品编号]}
def convert_to_atomic_files(args, train_data, valid_data, test_data, predict_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset, '\n')
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid] + valid_data[uid]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    # RecBole requires an item_id column; it is a placeholder for inference.
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.predict.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = predict_data[uid]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{item_seq[-1]}\n')
