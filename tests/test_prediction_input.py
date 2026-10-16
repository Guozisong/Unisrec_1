import tempfile
import unittest
from pathlib import Path

from data_pipeline.preprocessing.prediction_input import load_prediction_histories


class PredictionInputTest(unittest.TestCase):
    def test_uses_latest_known_items_in_time_order_for_new_users(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'prediction.csv'
            source.write_text('user_id,item_id,event_time\n'
                              'new-user,item-b,1791874162375\n'
                              'new-user,item-a,1791874162374\n'
                              'new-user,unknown,1791874162376\n'
                              'new-user,item-c,1791874162377\n'
                              'other-user,item-a,2026-01-01\n')
            histories, skipped = load_prediction_histories(
                source, {'item-a': 3, 'item-b': 5, 'item-c': 7}, max_seq_length=2)

        self.assertEqual(histories, [
            ('new-user', [5, 7]),
            ('other-user', [3]),
        ])
        self.assertEqual(skipped, 1)

    def test_fails_when_no_user_has_known_item_history(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'prediction.csv'
            source.write_text('user_id,item_id,event_time\n'
                              'new-user,unknown,1791874162374\n')
            with self.assertRaisesRegex(ValueError, 'No valid prediction histories'):
                load_prediction_histories(source, {'item-a': 3}, max_seq_length=50)
