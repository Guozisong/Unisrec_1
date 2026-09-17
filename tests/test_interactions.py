import csv
import tempfile
import unittest
from pathlib import Path

from data_pipeline.raw.prepare_interactions import prepare_csv


class InteractionContractTest(unittest.TestCase):
    def test_normalizes_column_order_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            target = Path(directory) / 'raw' / 'sample.csv'
            source.write_text('item_text,event_time,item_id,user_id,event_value,extra\n'
                              'Item A,2026-01-01,001,u1,2,ignored\n')
            prepare_csv(source, target)
            with target.open(newline='') as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows, [
                ['user_id', 'item_id', 'event_value', 'event_time', 'item_text'],
                ['u1', '001', '2', '2026-01-01', 'Item A'],
            ])

    def test_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            source.write_text('user_id,item_id\nu1,001\n')
            with self.assertRaisesRegex(ValueError, 'needs columns'):
                prepare_csv(source, Path(directory) / 'out.csv')

    def test_legacy_columns_are_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'legacy.csv'
            source.write_text('user_id,prod_id,purchase_count,dt,attrvalues\n'
                              'u1,001,2,2026-01-01,Item A\n')
            target = Path(directory) / 'normalized.csv'
            prepare_csv(source, target)
            self.assertEqual(target.read_text().splitlines(), [
                'user_id,item_id,event_value,event_time,item_text',
                'u1,001,2,2026-01-01,Item A',
            ])


if __name__ == '__main__':
    unittest.main()
