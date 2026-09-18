import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class SequenceSplitTest(unittest.TestCase):
    def test_eval_targets_are_held_out_and_prediction_uses_latest_history(self):
        torch_stub = types.ModuleType('torch')
        module_path = ROOT / 'data_pipeline/preprocessing/preprocessing_utils.py'
        spec = importlib.util.spec_from_file_location('preprocessing_utils_test', module_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch_stub}):
            spec.loader.exec_module(module)

        interactions = [('user', f'item{i}', 1.0, i) for i in range(6)]
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(dataset='sample', output_path=temporary, max_seq_length=4)
            train, valid, test, predict, users, items = module.generate_training_data(args, interactions)
            self.assertEqual([items[f'item{i}'] for i in range(2, 6)], [2, 3, 4, 5])
            self.assertEqual(train[users['user']], ['2', '3'])
            self.assertEqual(valid[users['user']], ['4'])
            self.assertEqual(test[users['user']], ['5'])
            self.assertEqual(predict[users['user']], ['2', '3', '4', '5'])

            os.mkdir(Path(temporary) / 'sample')
            module.convert_to_atomic_files(args, train, valid, test, predict)
            generated = Path(temporary) / 'sample'
            self.assertEqual((generated / 'sample.train.inter').read_text().splitlines()[1:],
                             ['0\t2\t3'])
            self.assertEqual((generated / 'sample.valid.inter').read_text().splitlines()[1:],
                             ['0\t2 3\t4'])
            self.assertEqual((generated / 'sample.test.inter').read_text().splitlines()[1:],
                             ['0\t2 3 4\t5'])
            self.assertEqual((generated / 'sample.predict.inter').read_text().splitlines()[1:],
                             ['0\t2 3 4 5\t5'])


if __name__ == '__main__':
    unittest.main()
