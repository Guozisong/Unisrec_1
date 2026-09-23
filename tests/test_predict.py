import csv
import importlib
import io
import math
import sys
import types
import unittest
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


class PredictionHelpersTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
