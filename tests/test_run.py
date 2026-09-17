import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_runs_stages_in_order_and_passes_generated_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            encoder = work / "encoder"
            encoder.mkdir()
            dataset = 'sample'
            input_csv = work / 'interactions.csv'
            input_csv.write_text('user_id,item_id,event_value,event_time,item_text\n')
            metadata_csv = work / 'item_metadata.csv'
            metadata_csv.write_text('item_id,category_id\n')
            eligible_csv = work / 'eligible_items.csv'
            eligible_csv.write_text('item_id\n')
            fake_python = work / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "stage = pathlib.Path(args[0]).name\n"
                "with open(os.environ['STAGE_LOG'], 'a') as log: log.write(' '.join(args) + '\\n')\n"
                "if stage == os.getenv('FAIL_STAGE'): sys.exit(12)\n"
                "def value(name): return args[args.index(name) + 1]\n"
                "if stage == 'prepare_interactions.py':\n"
                "    path = pathlib.Path(value('--output-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    (path / (value('--dataset') + '.csv')).touch()\n"
                "elif stage == 'preprocess.py':\n"
                "    path = pathlib.Path(value('--output_path')) / value('--dataset')\n"
                "    path.mkdir(parents=True, exist_ok=True)\n"
                "    for suffix in ('train.inter', 'valid.inter', 'test.inter', 'feat1CLS', 'feat2CLS'): (path / (value('--dataset') + '.' + suffix)).touch()\n"
                "    for name in ('index2user.json', 'index2item.json'): (path / name).touch()\n"
                "elif stage == 'pretrain.py':\n"
                "    path = pathlib.Path(value('--checkpoint-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    checkpoint = path / 'pretrained.pth'; checkpoint.touch()\n"
                "    pathlib.Path(value('--checkpoint-path-file')).write_text(str(checkpoint))\n"
                "elif stage == 'finetune.py':\n"
                "    path = pathlib.Path(value('--checkpoint-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    (path / ('UniSRec-' + value('-d') + '-finetuned.pth')).touch()\n"
            )
            fake_python.chmod(0o755)
            log = work / "stages.log"
            result = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "all", "--dataset", dataset, "--work-dir", str(work),
                 "--input-csv", str(input_csv), "--item-metadata-csv", str(metadata_csv),
                 "--eligible-items-csv", str(eligible_csv), "--plm-path", str(encoder),
                 "--python", str(fake_python), "--top-k", "30"],
                cwd=work,
                env={**os.environ, "STAGE_LOG": str(log)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = log.read_text().splitlines()
            self.assertEqual(
                [Path(line.split()[0]).name for line in lines],
                ["prepare_interactions.py", "preprocess.py", "pretrain.py", "finetune.py", "predict.py"],
            )
            self.assertIn(str(work / "downstream"), lines[2])
            self.assertIn(str(work / "checkpoints" / "pretrain" / "pretrained.pth"), lines[3])
            self.assertIn(str(work / "checkpoints" / "finetune" / "UniSRec-sample-finetuned.pth"), lines[4])
            self.assertIn(str(metadata_csv), lines[4])

            single_stages = [
                ("fetch", ["--input-csv", str(input_csv)], "prepare_interactions.py"),
                ("preprocess", ["--plm-path", str(encoder)], "preprocess.py"),
                ("pretrain", [], "pretrain.py"),
                ("finetune", ["--pretrained-checkpoint", str(work / "checkpoints" / "pretrain" / "pretrained.pth")], "finetune.py"),
                ("predict", ["--item-metadata-csv", str(metadata_csv), "--eligible-items-csv", str(eligible_csv), "--finetuned-checkpoint", str(work / "checkpoints" / "finetune" / "UniSRec-sample-finetuned.pth")], "predict.py"),
            ]
            for stage, options, expected in single_stages:
                with self.subTest(stage=stage):
                    log.write_text("")
                    single = subprocess.run(
                        ["bash", str(ROOT / "run.sh"), "--stage", stage, "--dataset", dataset, "--work-dir", str(work),
                         "--python", str(fake_python), *options],
                        cwd=work,
                        env={**os.environ, "STAGE_LOG": str(log)},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(single.returncode, 0, single.stderr)
                    self.assertEqual([Path(line.split()[0]).name for line in log.read_text().splitlines()],
                                     [expected])

            log.write_text("")
            failed = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--dataset", dataset, "--work-dir", str(work), "--plm-path", str(encoder),
                 "--input-csv", str(input_csv), "--item-metadata-csv", str(metadata_csv),
                 "--eligible-items-csv", str(eligible_csv), "--python", str(fake_python)],
                cwd=work,
                env={**os.environ, "STAGE_LOG": str(log), "FAIL_STAGE": "preprocess.py"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 12)
            self.assertEqual([Path(line.split()[0]).name for line in log.read_text().splitlines()],
                             ["prepare_interactions.py", "preprocess.py"])

            env_file = work / '.env'
            env_file.touch()
            query_file = work / 'interactions.sql'
            query_file.write_text('SELECT * FROM interactions')
            for option, value in (("--input-table", "interactions"),
                                  ("--input-query-file", str(query_file))):
                with self.subTest(source=option):
                    log.write_text('')
                    fetched = subprocess.run(
                        ["bash", str(ROOT / "run.sh"), "--stage", "fetch", "--dataset", dataset,
                         "--work-dir", str(work), "--env-file", str(env_file), "--python", str(fake_python),
                         option, value], cwd=work, env={**os.environ, "STAGE_LOG": str(log)},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(fetched.returncode, 0, fetched.stderr)
                    self.assertIn(option, log.read_text())


if __name__ == "__main__":
    unittest.main()
