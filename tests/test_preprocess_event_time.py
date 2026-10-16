import datetime
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class EventTimeTest(unittest.TestCase):
    def test_load_ratings_keeps_millisecond_order_and_accepts_dates(self):
        module_path = ROOT / 'data_pipeline/preprocessing/preprocess.py'
        spec = importlib.util.spec_from_file_location('preprocess_event_time_test', module_path)
        module = importlib.util.module_from_spec(spec)
        utils_stub = types.ModuleType('preprocessing_utils')
        runtime_stub = types.ModuleType('preprocessing_runtime')
        for name in ('filter_inters', 'make_inters_in_order', 'generate_training_data',
                     'generate_item_embedding', 'convert_to_atomic_files'):
            setattr(utils_stub, name, None)
        for name in ('check_path', 'set_device', 'load_plm'):
            setattr(runtime_stub, name, None)
        with patch.dict(sys.modules, {'preprocessing_utils': utils_stub,
                                      'preprocessing_runtime': runtime_stub}):
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'sample.csv'
            source.write_text('user_id,item_id,event_value,event_time,item_text\n'
                              'u1,i1,1,1791874162374,First\n'
                              'u1,i2,1,1791874162375,Second\n'
                              'u1,i3,1,2026-01-01,Third\n')
            interactions = module.load_ratings(source)

        times = {item: timestamp for _, item, _, timestamp in interactions}
        self.assertEqual(times['i1'], 1791874162374)
        self.assertEqual(times['i2'], 1791874162375)
        self.assertEqual(times['i3'],
                         int(datetime.datetime.strptime('2026-01-01', '%Y-%m-%d').timestamp() * 1000))
        self.assertLess(times['i1'], times['i2'])


if __name__ == '__main__':
    unittest.main()
