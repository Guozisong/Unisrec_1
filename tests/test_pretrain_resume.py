import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class PretrainResumeTest(unittest.TestCase):
    def test_restores_model_optimizer_and_next_epoch(self):
        torch = types.ModuleType('torch')
        checkpoint = {
            'config': {'dataset': 'sample'},
            'epoch': 11,
            'state_dict': {'weight': 'model-state'},
            'optimizer': {'momentum': 'optimizer-state'},
            'other_parameter': {'extra': 'model-extra'},
        }
        loaded = {}

        def load(path, map_location=None):
            loaded['path'] = path
            loaded['map_location'] = map_location
            return checkpoint

        torch.load = load
        recbole = types.ModuleType('recbole')
        recbole.config = types.ModuleType('recbole.config')
        recbole.config.Config = object
        recbole.trainer = types.ModuleType('recbole.trainer')
        recbole.trainer.trainer = types.ModuleType('recbole.trainer.trainer')
        recbole.trainer.trainer.PretrainTrainer = object
        recbole.utils = types.ModuleType('recbole.utils')
        recbole.utils.init_seed = lambda *args: None
        recbole.utils.init_logger = lambda *args: None
        unisrec = types.ModuleType('unisrec')
        unisrec.UniSRec = object
        dataset_module = types.ModuleType('recbole_data.dataset')
        dataset_module.UniSRecDataset = object
        dataloader_module = types.ModuleType('recbole_data.dataloader')
        dataloader_module.CustomizedTrainDataLoader = object
        modules = {
            'torch': torch,
            'recbole': recbole,
            'recbole.config': recbole.config,
            'recbole.trainer': recbole.trainer,
            'recbole.trainer.trainer': recbole.trainer.trainer,
            'recbole.utils': recbole.utils,
            'unisrec': unisrec,
            'recbole_data.dataset': dataset_module,
            'recbole_data.dataloader': dataloader_module,
        }
        spec = importlib.util.spec_from_file_location('pretrain_resume_test', ROOT / 'pretrain.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, 'restore_pretrain_checkpoint'))

        class Recorder:
            def __init__(self):
                self.value = None

            def load_state_dict(self, value):
                self.value = value

        model = Recorder()
        model.load_other_parameter = lambda value: setattr(model, 'other', value)
        trainer = types.SimpleNamespace(optimizer=Recorder(), start_epoch=0)
        config = {'dataset': 'sample', 'device': 'cuda', 'pretrain_epochs': 50}

        module.restore_pretrain_checkpoint('epoch-12.pth', config, model, trainer)

        self.assertEqual(loaded, {'path': 'epoch-12.pth', 'map_location': 'cuda'})
        self.assertEqual(model.value, {'weight': 'model-state'})
        self.assertEqual(model.other, {'extra': 'model-extra'})
        self.assertEqual(trainer.optimizer.value, {'momentum': 'optimizer-state'})
        self.assertEqual(trainer.start_epoch, 12)


if __name__ == '__main__':
    unittest.main()
