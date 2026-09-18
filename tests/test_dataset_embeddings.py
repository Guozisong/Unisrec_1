import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DatasetEmbeddingPathTest(unittest.TestCase):
    def test_loads_embeddings_from_recbole_dataset_path(self):
        torch = types.ModuleType('torch')
        torch.nn = types.ModuleType('torch.nn')
        torch.nn.Embedding = object
        recbole = types.ModuleType('recbole')
        recbole.data = types.ModuleType('recbole.data')
        recbole.data.dataset = types.ModuleType('recbole.data.dataset')
        recbole.data.dataset.SequentialDataset = object
        modules = {
            'torch': torch, 'torch.nn': torch.nn, 'recbole': recbole,
            'recbole.data': recbole.data, 'recbole.data.dataset': recbole.data.dataset,
        }
        spec = importlib.util.spec_from_file_location('dataset_path_test', ROOT / 'recbole_data/dataset.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / 'sample'
            dataset_dir.mkdir()
            first = np.array([[0, 0], [1, 2], [3, 4]], dtype=np.float32)
            second = np.array([[0, 0], [5, 6], [7, 8]], dtype=np.float32)
            first.tofile(dataset_dir / 'sample.feat1CLS')
            second.tofile(dataset_dir / 'sample.feat2CLS')

            dataset = module.UniSRecDataset.__new__(module.UniSRecDataset)
            dataset.config = {'data_path': str(dataset_dir)}
            dataset.dataset_name = 'sample'
            dataset.plm_size = 2
            dataset.plm_suffix = 'feat1CLS'
            dataset.plm_suffix_aug = 'feat2CLS'
            dataset.item_num = 3
            dataset.field2id_token = {'item_id': ['[PAD]', '1', '2']}

            np.testing.assert_array_equal(dataset.load_plm_embedding1(), first)
            np.testing.assert_array_equal(dataset.load_plm_embedding2(), second)


if __name__ == '__main__':
    unittest.main()
