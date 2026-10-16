import csv
import contextlib
import importlib
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def import_predict():
    modules = {
        'torch': types.ModuleType('torch'),
        'recbole': types.ModuleType('recbole'),
        'recbole.config': types.ModuleType('recbole.config'),
        'recbole.data': types.ModuleType('recbole.data'),
        'unisrec': types.ModuleType('unisrec'),
        'recbole_data': types.ModuleType('recbole_data'),
        'recbole_data.dataset': types.ModuleType('recbole_data.dataset'),
    }
    modules['recbole.config'].Config = object
    modules['recbole.data'].data_preparation = object
    modules['unisrec'].UniSRec = object
    modules['recbole_data.dataset'].UniSRecDataset = object
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop('predict', None)
        return importlib.import_module('predict')


class FakeScores:
    def __init__(self):
        self.call = None
        self.assignments = []

    def __setitem__(self, key, value):
        self.assignments.append((key, value))

    def scatter_(self, dimension, indices, value):
        self.call = (dimension, indices, value)
        return self


class FakeTensor:
    def __init__(self, values, **kwargs):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class PredictionHelpersTest(unittest.TestCase):
    def test_pads_current_histories_for_model_without_reindexing_items(self):
        predict = import_predict()

        users, sequences, lengths = predict.pad_prediction_batch(
            [('new-user', [5, 3]), ('another-user', [7])], max_length=4)

        self.assertEqual(users, ['new-user', 'another-user'])
        self.assertEqual(sequences, [[5, 3, 0, 0], [7, 0, 0, 0]])
        self.assertEqual(lengths, [2, 1])

    def test_mask_history_scores_uses_sequence_ids_including_padding(self):
        predict = import_predict()
        scores = FakeScores()
        sequence_ids = [[4, 9, 0], [3, 0, 0]]

        returned = predict.mask_history_scores(scores, sequence_ids)

        self.assertIs(returned, scores)
        self.assertEqual(scores.call[:2], (1, sequence_ids))
        self.assertTrue(math.isinf(scores.call[2]))
        self.assertLess(scores.call[2], 0)
        self.assertEqual(scores.assignments[0][0], (slice(None), 0))
        self.assertTrue(math.isinf(scores.assignments[0][1]))
        self.assertLess(scores.assignments[0][1], 0)

    def test_build_original_id_lookup_maps_recbole_tokens_once(self):
        predict = import_predict()

        lookup = predict.build_original_id_lookup(
            ['[PAD]', '11', '7'], {'11': 'user-a', '7': 'user-b'}
        )

        self.assertEqual(lookup, [None, 'user-a', 'user-b'])

    def test_write_prediction_rows_writes_only_finite_scores(self):
        predict = import_predict()
        output = io.StringIO()
        writer = csv.writer(output, lineterminator='\n')

        count = predict.write_prediction_rows(
            writer,
            user_ids=[1, 2],
            item_ids=[[2, 3], [1, 2]],
            scores=[[0.9, float('-inf')], [0.8, 0.7]],
            user_lookup=[None, 'u1', 'u2'],
            item_lookup=[None, 'i1', 'i2', 'i3'],
        )

        self.assertEqual(count, 3)
        self.assertEqual(output.getvalue().splitlines(), [
            'u1,i2,0.9',
            'u2,i1,0.8',
            'u2,i2,0.7',
        ])

    def test_predicts_new_users_with_frozen_training_item_ids(self):
        predict = import_predict()
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / 'sample'
            dataset_dir.mkdir()
            (dataset_dir / 'index2item.json').write_text(json.dumps({
                '0': 'item-a', '1': 'item-b', '2': 'item-c'}))
            interactions = Path(temporary) / 'current.csv'
            interactions.write_text('user_id,item_id,event_time\n'
                                    'new-user,item-a,1791874162374\n'
                                    'another-user,item-b,1791874162375\n')
            model = mock.Mock()
            model.to.return_value = model
            model.get_full_sort_item_embeddings.return_value = 'candidate-vectors'
            dataset = types.SimpleNamespace(
                field2id_token={'item_id': ['[PAD]', '0', '1', '2']}, item_num=4)
            config = {
                'device': 'cpu', 'data_path': str(dataset_dir),
                'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id',
                'LIST_SUFFIX': '_list', 'ITEM_LIST_LENGTH_FIELD': 'item_length',
                'MAX_ITEM_LIST_LENGTH': 4, 'eval_batch_size': 2,
            }
            with mock.patch.object(predict, 'Config', return_value=config) as config_factory, \
                 mock.patch.object(predict, 'UniSRecDataset', return_value=dataset), \
                 mock.patch.object(predict, 'data_preparation', return_value=(
                     None, None, types.SimpleNamespace(dataset=dataset))), \
                 mock.patch.object(predict, 'UniSRec', return_value=model), \
                 mock.patch.object(predict, 'mask_history_scores'), \
                 mock.patch.object(predict.torch, 'load', return_value={
                     'state_dict': {}, 'other_parameter': None}, create=True), \
                 mock.patch.object(predict.torch, 'tensor', side_effect=FakeTensor, create=True), \
                 mock.patch.object(predict.torch, 'long', 'long', create=True), \
                 mock.patch.object(predict.torch, 'inference_mode', side_effect=contextlib.nullcontext,
                                   create=True), \
                 mock.patch.object(predict.torch, 'topk', return_value=(
                     FakeTensor([[0.9], [0.8]]), FakeTensor([[3], [3]])), create=True):
                output = predict.predictor('sample', 'model.pth', 1, temporary,
                                           data_path=temporary, interactions_file=interactions,
                                           max_seq_length=2)

            self.assertEqual(config_factory.call_args.kwargs['config_dict']['benchmark_filename'],
                             ['train', 'valid', 'test'])
            interaction, candidate_vectors = model.full_sort_predict_with_item_embeddings.call_args.args
            self.assertEqual(interaction['item_id_list'].values,
                             [[1, 0, 0, 0], [2, 0, 0, 0]])
            self.assertEqual(interaction['item_length'].values, [1, 1])
            self.assertEqual(candidate_vectors, 'candidate-vectors')
            self.assertEqual(Path(output).read_text().splitlines(), [
                'user_id,item_id,score',
                'new-user,item-c,0.9',
                'another-user,item-c,0.8',
            ])


if __name__ == '__main__':
    unittest.main()
