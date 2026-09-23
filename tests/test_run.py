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
                "    for suffix in ('train.inter', 'valid.inter', 'test.inter', 'predict.inter', 'feat1CLS', 'feat2CLS'): (path / (value('--dataset') + '.' + suffix)).touch()\n"
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
                 "--input-csv", str(input_csv), "--plm-path", str(encoder),
                 "--python", str(fake_python), "--top-k", "30", "--max-seq-length", "40"],
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
            for stage_name in ('fetch', 'preprocess', 'pretrain', 'finetune', 'predict'):
                self.assertIn(f'stage={stage_name} START dataset=sample work_dir={work}', result.stdout)
                self.assertIn(f'stage={stage_name} COMPLETE elapsed=', result.stdout)
            self.assertLess(result.stdout.index('stage=fetch COMPLETE'),
                            result.stdout.index('stage=preprocess START'))
            self.assertIn(str(work / "downstream"), lines[2])
            self.assertIn('--max_seq_length 40', lines[1])
            self.assertIn(str(work / "checkpoints" / "pretrain" / "pretrained.pth"), lines[3])
            self.assertIn(str(work / "checkpoints" / "finetune" / "UniSRec-sample-finetuned.pth"), lines[4])
            self.assertIn('--data-path', lines[4])
            self.assertIn('-t 30', lines[4])
            self.assertNotIn('--item-metadata', lines[4])
            self.assertNotIn('--eligible-items', lines[4])
            self.assertNotIn('--output-table', lines[4])

            single_stages = [
                ("fetch", ["--input-csv", str(input_csv)], "prepare_interactions.py"),
                ("preprocess", ["--plm-path", str(encoder), "--max-seq-length", "40"], "preprocess.py"),
                ("pretrain", [], "pretrain.py"),
                ("finetune", ["--pretrained-checkpoint", str(work / "checkpoints" / "pretrain" / "pretrained.pth")], "finetune.py"),
                ("predict", ["--finetuned-checkpoint", str(work / "checkpoints" / "finetune" / "UniSRec-sample-finetuned.pth")], "predict.py"),
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
                    self.assertIn(f'stage={stage} START', single.stdout)
                    self.assertIn(f'stage={stage} COMPLETE', single.stdout)
                    if stage == 'preprocess':
                        self.assertIn('--max_seq_length 40', log.read_text())

            resume_checkpoint = work / 'checkpoints' / 'pretrain' / 'epoch-12.pth'
            resume_checkpoint.touch()
            log.write_text('')
            resumed = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "pretrain", "--dataset", dataset,
                 "--work-dir", str(work), "--python", str(fake_python),
                 "--resume-checkpoint", str(resume_checkpoint)],
                cwd=work,
                env={**os.environ, "STAGE_LOG": str(log)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn(f'--resume-checkpoint {resume_checkpoint}', log.read_text())

            invalid = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "preprocess", "--dataset", dataset,
                 "--work-dir", str(work), "--plm-path", str(encoder), "--python", str(fake_python),
                 "--max-seq-length", "101"],
                cwd=work, capture_output=True, text=True,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn('--max-seq-length', invalid.stderr)

            log.write_text("")
            failed = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--dataset", dataset, "--work-dir", str(work), "--plm-path", str(encoder),
                 "--input-csv", str(input_csv), "--python", str(fake_python)],
                cwd=work,
                env={**os.environ, "STAGE_LOG": str(log), "FAIL_STAGE": "preprocess.py"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 12)
            self.assertIn('stage=preprocess START', failed.stdout)
            self.assertNotIn('stage=preprocess COMPLETE', failed.stdout)
            self.assertIn('stage=preprocess FAILED exit=12', failed.stderr)
            self.assertEqual([Path(line.split()[0]).name for line in log.read_text().splitlines()],
                             ["prepare_interactions.py", "preprocess.py"])

            failed_pretrain = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "pretrain", "--dataset", dataset,
                 "--work-dir", str(work), "--python", str(fake_python)],
                cwd=work, env={**os.environ, "STAGE_LOG": str(log), "FAIL_STAGE": "pretrain.py"},
                capture_output=True, text=True,
            )
            self.assertEqual(failed_pretrain.returncode, 12)
            self.assertIn('stage=pretrain FAILED exit=12', failed_pretrain.stderr)
            self.assertEqual(list(work.glob('.pretrain-path.*')), [])

            removed = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "predict", "--dataset", dataset,
                 "--work-dir", str(work), "--python", str(fake_python),
                 "--item-metadata-csv", "items.csv"],
                cwd=work, capture_output=True, text=True,
            )
            self.assertEqual(removed.returncode, 2)
            self.assertIn('Unknown option: --item-metadata-csv', removed.stderr)

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
