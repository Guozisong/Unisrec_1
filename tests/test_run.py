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
            env_file = work / ".env"
            env_file.touch()
            fake_python = work / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "stage = pathlib.Path(args[0]).name\n"
                "with open(os.environ['STAGE_LOG'], 'a') as log: log.write(' '.join(args) + '\\n')\n"
                "if stage == os.getenv('FAIL_STAGE'): sys.exit(12)\n"
                "def value(name): return args[args.index(name) + 1]\n"
                "if stage == 'get_data_from_odps.py':\n"
                "    path = pathlib.Path(value('--output-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    (path / 'lianhua.csv').touch()\n"
                "elif stage == 'process_or.py':\n"
                "    path = pathlib.Path(value('--output_path')) / value('--dataset')\n"
                "    path.mkdir(parents=True, exist_ok=True)\n"
                "    for suffix in ('train.inter', 'valid.inter', 'test.inter', 'feat1CLS', 'feat2CLS'): (path / ('lianhua.' + suffix)).touch()\n"
                "    for name in ('index2user.json', 'index2item.json'): (path / name).touch()\n"
                "elif stage == 'pretrain.py':\n"
                "    path = pathlib.Path(value('--checkpoint-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    checkpoint = path / 'pretrained.pth'; checkpoint.touch()\n"
                "    pathlib.Path(value('--checkpoint-path-file')).write_text(str(checkpoint))\n"
                "elif stage == 'finetune.py':\n"
                "    path = pathlib.Path(value('--checkpoint-dir')); path.mkdir(parents=True, exist_ok=True)\n"
                "    (path / 'UniSRec-lianhua-finetuned.pth').touch()\n"
            )
            fake_python.chmod(0o755)
            log = work / "stages.log"
            result = subprocess.run(
                ["bash", str(ROOT / "run.sh"), "--stage", "all", "--dataset", "lianhua", "--work-dir", str(work),
                 "--plm-path", str(encoder), "--env-file", str(env_file),
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
                ["get_data_from_odps.py", "process_or.py", "pretrain.py", "finetune.py", "predict.py"],
            )
            self.assertIn(str(work / "downstream"), lines[2])
            self.assertIn(str(work / "checkpoints" / "pretrain" / "pretrained.pth"), lines[3])
            self.assertIn(str(work / "checkpoints" / "finetune" / "UniSRec-lianhua-finetuned.pth"), lines[4])

            single_stages = [
                ("fetch", ["--env-file", str(env_file)], "get_data_from_odps.py"),
                ("preprocess", ["--plm-path", str(encoder)], "process_or.py"),
                ("pretrain", [], "pretrain.py"),
                ("finetune", ["--pretrained-checkpoint", str(work / "checkpoints" / "pretrain" / "pretrained.pth")], "finetune.py"),
                ("predict", ["--env-file", str(env_file), "--finetuned-checkpoint", str(work / "checkpoints" / "finetune" / "UniSRec-lianhua-finetuned.pth")], "predict.py"),
            ]
            for stage, options, expected in single_stages:
                with self.subTest(stage=stage):
                    log.write_text("")
                    single = subprocess.run(
                        ["bash", str(ROOT / "run.sh"), "--stage", stage, "--work-dir", str(work),
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
                ["bash", str(ROOT / "run.sh"), "--work-dir", str(work), "--plm-path", str(encoder),
                 "--env-file", str(env_file), "--python", str(fake_python)],
                cwd=work,
                env={**os.environ, "STAGE_LOG": str(log), "FAIL_STAGE": "process_or.py"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 12)
            self.assertEqual([Path(line.split()[0]).name for line in log.read_text().splitlines()],
                             ["get_data_from_odps.py", "process_or.py"])


if __name__ == "__main__":
    unittest.main()
