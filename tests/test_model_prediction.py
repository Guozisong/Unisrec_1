import importlib
import sys
import types
import unittest
from unittest import mock


def import_unisrec():
    torch = types.ModuleType('torch')
    torch_nn = types.ModuleType('torch.nn')
    torch_functional = types.ModuleType('torch.nn.functional')
    sasrec = types.ModuleType('recbole.model.sequential_recommender.sasrec')

    class Module:
        pass

    class SASRec(Module):
        pass

    torch_nn.Module = Module
    torch_nn.Parameter = object
    torch_nn.Dropout = object
    torch_nn.Linear = object
    torch_nn.ModuleList = list
    torch.nn = torch_nn
    sasrec.SASRec = SASRec

    modules = {
        'torch': torch,
        'torch.nn': torch_nn,
        'torch.nn.functional': torch_functional,
        'recbole': types.ModuleType('recbole'),
        'recbole.model': types.ModuleType('recbole.model'),
        'recbole.model.sequential_recommender': types.ModuleType(
            'recbole.model.sequential_recommender'),
        'recbole.model.sequential_recommender.sasrec': sasrec,
    }
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop('unisrec', None)
        return importlib.import_module('unisrec')


class CachedScoringTest(unittest.TestCase):
    def test_builds_and_normalizes_transductive_item_embeddings(self):
        unisrec = import_unisrec()
        model = object.__new__(unisrec.UniSRec)
        model.plm_embedding = types.SimpleNamespace(weight=10)
        model.moe_adaptor = mock.Mock(return_value=20)
        model.train_stage = 'transductive_ft'
        model.item_embedding = types.SimpleNamespace(weight=3)
        unisrec.F.normalize = mock.Mock(return_value='normalized')

        result = model.get_full_sort_item_embeddings()

        self.assertEqual(result, 'normalized')
        model.moe_adaptor.assert_called_once_with(10)
        unisrec.F.normalize.assert_called_once_with(23, dim=-1)

    def test_scores_a_batch_with_supplied_item_embeddings(self):
        unisrec = import_unisrec()
        model = object.__new__(unisrec.UniSRec)
        model.ITEM_SEQ = 'item_seq'
        model.ITEM_SEQ_LEN = 'item_seq_len'
        model.plm_embedding = mock.Mock(return_value='raw-sequence')
        model.moe_adaptor = mock.Mock(return_value='adapted-sequence')
        model.forward = mock.Mock(return_value='sequence-output')
        item_embeddings = mock.Mock()
        item_embeddings.transpose.return_value = 'transposed-items'
        unisrec.F.normalize = mock.Mock(return_value='normalized-sequence')
        unisrec.torch.matmul = mock.Mock(return_value='scores')

        result = model.full_sort_predict_with_item_embeddings(
            {'item_seq': 'sequence-ids', 'item_seq_len': 'lengths'},
            item_embeddings,
        )

        self.assertEqual(result, 'scores')
        model.plm_embedding.assert_called_once_with('sequence-ids')
        model.forward.assert_called_once_with(
            'sequence-ids', 'adapted-sequence', 'lengths')
        item_embeddings.transpose.assert_called_once_with(0, 1)
        unisrec.torch.matmul.assert_called_once_with(
            'normalized-sequence', 'transposed-items')

    def test_existing_full_sort_predict_delegates_to_cached_scoring(self):
        unisrec = import_unisrec()
        model = object.__new__(unisrec.UniSRec)
        model.get_full_sort_item_embeddings = mock.Mock(return_value='items')
        model.full_sort_predict_with_item_embeddings = mock.Mock(return_value='scores')
        interaction = object()

        result = model.full_sort_predict(interaction)

        self.assertEqual(result, 'scores')
        model.get_full_sort_item_embeddings.assert_called_once_with()
        model.full_sort_predict_with_item_embeddings.assert_called_once_with(
            interaction, 'items')


if __name__ == '__main__':
    unittest.main()
