import argparse
import csv
import json
import math
import os

import torch
from recbole.config import Config
from recbole.data import data_preparation

from data_pipeline.preprocessing.prediction_input import load_prediction_histories
from recbole_data.dataset import UniSRecDataset
from unisrec import UniSRec


def mask_history_scores(scores, item_sequences):
    scores[:, 0] = float('-inf')
    return scores.scatter_(1, item_sequences, float('-inf'))


def build_original_id_lookup(id_tokens, original_ids):
    return [
        None if str(token) == '[PAD]' else original_ids[str(token)]
        for token in id_tokens
    ]


def pad_prediction_batch(histories, max_length):
    users = [user_id for user_id, _ in histories]
    sequences = [item_ids[-max_length:] for _, item_ids in histories]
    lengths = [len(item_ids) for item_ids in sequences]
    return users, [item_ids + [0] * (max_length - len(item_ids)) for item_ids in sequences], lengths


def write_prediction_rows(writer, user_ids, item_ids, scores,
                          user_lookup, item_lookup):
    row_count = 0
    for user_id, recommended_items, recommended_scores in zip(
            user_ids, item_ids, scores):
        original_user_id = user_lookup[int(user_id)]
        for item_id, score in zip(recommended_items, recommended_scores):
            score = float(score)
            if not math.isfinite(score):
                continue
            writer.writerow((original_user_id, item_lookup[int(item_id)], score))
            row_count += 1
    return row_count


def predictor(dataset, model_file, top_k, result_save_path, data_path=None,
              interactions_file=None, max_seq_length=50):
    config_files = ['configs/UniSRec.yaml', 'configs/finetune.yaml']
    benchmark_files = ['train', 'valid', 'test'] if interactions_file else ['train', 'valid', 'predict']
    overrides = {'benchmark_filename': benchmark_files}
    if data_path:
        overrides['data_path'] = data_path
    config = Config(
        model=UniSRec,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=overrides,
    )
    device = config['device']

    print('[predict] Loading dataset')
    dataset_object = UniSRecDataset(config)
    _, _, prediction_data = data_preparation(config, dataset_object)

    dataset_path = config['data_path']
    with open(os.path.join(dataset_path, 'index2item.json'), encoding='utf-8') as file:
        index2item = json.load(file)

    if interactions_file is None:
        with open(os.path.join(dataset_path, 'index2user.json'), encoding='utf-8') as file:
            index2user = json.load(file)
        user_lookup = build_original_id_lookup(
            dataset_object.field2id_token[config['USER_ID_FIELD']], index2user)
    item_lookup = build_original_id_lookup(
        dataset_object.field2id_token[config['ITEM_ID_FIELD']], index2item)
    if interactions_file is not None:
        if max_seq_length > config['MAX_ITEM_LIST_LENGTH']:
            raise ValueError('--max-seq-length exceeds the model sequence length limit')
        item_ids = {original_id: model_id for model_id, original_id in enumerate(item_lookup)
                    if original_id is not None}
        histories, skipped = load_prediction_histories(
            interactions_file, item_ids, max_seq_length)
        print(f'[predict] Loaded {len(histories)} users; skipped {skipped} invalid or unknown-item interactions')

    print('[predict] Loading model')
    model = UniSRec(config, prediction_data.dataset).to(device)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    model.eval()

    os.makedirs(result_save_path, exist_ok=True)
    output_path = os.path.join(
        result_save_path, f'{dataset}-recommendations.csv')
    total_items = prediction_data.dataset.item_num
    recommendation_count = min(top_k, max(total_items - 1, 0))
    row_count = 0

    print('[predict] Generating recommendations')
    with torch.inference_mode():
        item_embeddings = model.get_full_sort_item_embeddings()
        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file, lineterminator='\n')
            writer.writerow(('user_id', 'item_id', 'score'))

            if interactions_file is None:
                batches = prediction_data
            else:
                batch_size = config['eval_batch_size']
                batches = (histories[start:start + batch_size]
                           for start in range(0, len(histories), batch_size))
            for batch in batches:
                if interactions_file is None:
                    interaction = batch[0].to(device)
                    user_ids = interaction[config['USER_ID_FIELD']].detach().cpu().tolist()
                    current_user_lookup = user_lookup
                else:
                    current_user_lookup, sequences, lengths = pad_prediction_batch(
                        batch, config['MAX_ITEM_LIST_LENGTH'])
                    interaction = {
                        config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']: torch.tensor(
                            sequences, dtype=torch.long, device=device),
                        config['ITEM_LIST_LENGTH_FIELD']: torch.tensor(
                            lengths, dtype=torch.long, device=device),
                    }
                    user_ids = range(len(current_user_lookup))
                item_sequences = interaction[
                    config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']]
                scores = model.full_sort_predict_with_item_embeddings(
                    interaction, item_embeddings)
                mask_history_scores(scores, item_sequences)
                if recommendation_count == 0:
                    continue
                topk_scores, topk_items = torch.topk(
                    scores, k=recommendation_count, dim=1)
                row_count += write_prediction_rows(
                    writer,
                    user_ids,
                    topk_items.detach().cpu().tolist(),
                    topk_scores.detach().cpu().tolist(),
                    current_user_lookup,
                    item_lookup,
                )

    print(f'[predict] Saved {row_count} rows to {output_path}')
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=True, help='dataset name')
    parser.add_argument('-fp', required=True, help='fine-tuned model path')
    parser.add_argument(
        '-t', type=int, default=50,
        help='number of recommendations per user')
    parser.add_argument(
        '-sp', default='outputs/results/', help='result directory')
    parser.add_argument('--data-path', help='parent directory of the dataset')
    parser.add_argument('--interactions-file', help='current user interaction CSV for repeated prediction')
    parser.add_argument('--max-seq-length', type=int, default=50,
                        help='recent interactions retained per user (1-100)')
    args = parser.parse_args()
    if args.t < 1:
        parser.error('-t must be a positive integer')
    if not 1 <= args.max_seq_length <= 100:
        parser.error('--max-seq-length must be from 1 to 100')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    predictor(
        arguments.d,
        arguments.fp,
        arguments.t,
        arguments.sp,
        arguments.data_path,
        arguments.interactions_file,
        arguments.max_seq_length,
    )
