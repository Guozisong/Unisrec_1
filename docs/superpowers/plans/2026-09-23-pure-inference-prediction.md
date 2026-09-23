# Pure Inference Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current metadata-filtered prediction job with efficient model-only TopK inference that streams one `user_id,item_id,score` CSV.

**Architecture:** `UniSRec` will expose one method for normalized all-item embeddings and one method for scoring a user batch with that cache, while its existing `full_sort_predict()` contract stays intact. `predict.py` will cache embeddings once, mask padding and sequence history on the active device, select only the requested TopK, map internal IDs once, and stream rows directly to CSV. `run.sh` and README will expose only the inputs the pure inference stage actually uses.

**Tech Stack:** Python 3, PyTorch, RecBole, standard-library `csv`, Bash, `unittest`

---

## File map

- Modify `tests/test_run.py`: lock the simplified `all` and `predict` Bash interfaces.
- Create `tests/test_predict.py`: verify history masking, internal-to-original ID mapping, finite-row CSV writing, and the command-line contract without importing real PyTorch or RecBole.
- Modify `unisrec.py`: add cached item-embedding construction and cached full-sort scoring while preserving `full_sort_predict()`.
- Rewrite `predict.py`: remove ODPS, pandas, category filtering, details output, and whole-dataset accumulation; implement one-pass inference and CSV streaming.
- Modify `run.sh`: remove obsolete prediction source/output arguments and invoke the new `predict.py` interface.
- Modify `README.md`: document the model-only prediction data flow, parameters, output, and standalone/full-pipeline examples.

### Task 1: Lock the simplified Bash interface

**Files:**
- Modify: `tests/test_run.py:12-171`
- Test: `tests/test_run.py`

- [ ] **Step 1: Change the full-pipeline fixture so prediction has no metadata inputs**

Delete creation of `item_metadata.csv` and `eligible_items.csv`, invoke `run.sh --stage all` with only the interaction source, encoder, work directory, Python executable, TopK, and sequence length, then assert the logged prediction command contains the checkpoint, `--data-path`, and `-t 30`:

```python
result = subprocess.run(
    ["bash", str(ROOT / "run.sh"), "--stage", "all", "--dataset", dataset,
     "--work-dir", str(work), "--input-csv", str(input_csv),
     "--plm-path", str(encoder), "--python", str(fake_python),
     "--top-k", "30", "--max-seq-length", "40"],
    cwd=work,
    env={**os.environ, "STAGE_LOG": str(log)},
    capture_output=True,
    text=True,
)
```

Replace the metadata assertion with:

```python
self.assertIn('--data-path', lines[4])
self.assertIn('-t 30', lines[4])
self.assertNotIn('--item-metadata', lines[4])
self.assertNotIn('--eligible-items', lines[4])
self.assertNotIn('--output-table', lines[4])
```

- [ ] **Step 2: Change standalone prediction and failure fixtures**

Use only the checkpoint for standalone prediction:

```python
("predict", ["--finetuned-checkpoint", str(
    work / "checkpoints" / "finetune" / "UniSRec-sample-finetuned.pth"
)], "predict.py"),
```

Remove metadata arguments from the intentional pipeline-failure invocation.

- [ ] **Step 3: Add a regression assertion for removed options**

Add a subprocess call using one removed option and assert it fails through the standard unknown-option path:

```python
removed = subprocess.run(
    ["bash", str(ROOT / "run.sh"), "--stage", "predict", "--dataset", dataset,
     "--work-dir", str(work), "--python", str(fake_python),
     "--item-metadata-csv", "items.csv"],
    cwd=work, capture_output=True, text=True,
)
self.assertEqual(removed.returncode, 2)
self.assertIn("Unknown option: --item-metadata-csv", removed.stderr)
```

- [ ] **Step 4: Run the test and confirm it fails for the old interface**

Run: `python3 -m unittest tests.test_run -v`

Expected: FAIL because `run.sh` still requires metadata and still accepts removed prediction options.

- [ ] **Step 5: Commit the failing interface test**

```bash
git add tests/test_run.py
git commit -m "test: define pure prediction pipeline interface"
```

### Task 2: Simplify `run.sh`

**Files:**
- Modify: `run.sh:4-280`
- Test: `tests/test_run.py`

- [ ] **Step 1: Remove obsolete prediction options and variables**

Delete these usage lines, shell variables, parser branches, validations, path normalization entries, and ODPS credential conditions:

```text
--item-metadata-csv
--item-metadata-table
--eligible-items-csv
--eligible-items-table
--output-table
```

Keep `--env-file` because `fetch` still uses it for `--input-table` and `--input-query-file`. Its existence check becomes:

```bash
if [[ ( "$stage" == all || "$stage" == fetch ) &&
      ( -n "$input_table" || -n "$input_query_file" ) ]]; then
  if [[ ! -f "$env_file" ]]; then
    echo "ODPS credentials file does not exist: $env_file" >&2
    exit 2
  fi
  env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")
fi
```

Normalize only fetch file sources:

```bash
for source in input_csv input_query_file; do
  if [[ -n "${!source}" ]]; then
    printf -v "$source" '%s/%s' "$(cd "$(dirname "${!source}")" && pwd)" "$(basename "${!source}")"
  fi
done
```

- [ ] **Step 2: Replace the prediction command construction**

Keep existing input and checkpoint checks, then invoke only the model inference arguments:

```bash
mkdir -p "$result_dir"
"$python" predict.py -d "$dataset" -fp "$checkpoint" -t "$top_k" \
  -sp "$result_dir" --data-path "$data_dir"
```

- [ ] **Step 3: Validate the Bash syntax and interface tests**

Run: `bash -n run.sh && python3 -m unittest tests.test_run -v`

Expected: Bash syntax succeeds and all `PipelineTest` cases pass.

- [ ] **Step 4: Commit the Bash change**

```bash
git add run.sh
git commit -m "refactor: simplify prediction pipeline arguments"
```

### Task 3: Define pure prediction behavior with tests

**Files:**
- Create: `tests/test_predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Add dependency stubs and import the prediction module**

Create a test that temporarily supplies minimal modules before importing `predict.py`, so local tests remain runnable without PyTorch or RecBole:

```python
import csv
import importlib
import io
import math
import sys
import types
import unittest
from unittest import mock


def import_predict():
    modules = {
        'torch': types.ModuleType('torch'),
        'recbole': types.ModuleType('recbole'),
        'recbole.config': types.ModuleType('recbole.config'),
        'recbole.data': types.ModuleType('recbole.data'),
        'unisrec': types.ModuleType('unisrec'),
        'recbole_data': types.ModuleType('recbole_data'),
        'recbole_data.dataset': types.ModuleType('recbole_data.dataset'),
    }
    modules['recbole.config'].Config = object
    modules['recbole.data'].data_preparation = object
    modules['unisrec'].UniSRec = object
    modules['recbole_data.dataset'].UniSRecDataset = object
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop('predict', None)
        return importlib.import_module('predict')
```

- [ ] **Step 2: Test that one scatter masks history and padding**

```python
class FakeScores:
    def __init__(self):
        self.call = None

    def scatter_(self, dimension, indices, value):
        self.call = (dimension, indices, value)
        return self


class PredictionHelpersTest(unittest.TestCase):
    def test_mask_history_scores_uses_sequence_ids_including_padding(self):
        predict = import_predict()
        scores = FakeScores()
        sequence_ids = [[4, 9, 0], [3, 0, 0]]

        returned = predict.mask_history_scores(scores, sequence_ids)

        self.assertIs(returned, scores)
        self.assertEqual(scores.call[:2], (1, sequence_ids))
        self.assertTrue(math.isinf(scores.call[2]))
        self.assertLess(scores.call[2], 0)
```

- [ ] **Step 3: Test ID lookup construction and finite CSV rows**

```python
def test_build_original_id_lookup_maps_recbole_tokens_once(self):
    predict = import_predict()
    lookup = predict.build_original_id_lookup(
        ['[PAD]', '11', '7'], {'11': 'user-a', '7': 'user-b'}
    )
    self.assertEqual(lookup, [None, 'user-a', 'user-b'])

def test_write_prediction_rows_writes_only_finite_scores(self):
    predict = import_predict()
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')

    count = predict.write_prediction_rows(
        writer,
        user_ids=[1, 2],
        item_ids=[[2, 3], [1, 2]],
        scores=[[0.9, float('-inf')], [0.8, 0.7]],
        user_lookup=[None, 'u1', 'u2'],
        item_lookup=[None, 'i1', 'i2', 'i3'],
    )

    self.assertEqual(count, 3)
    self.assertEqual(output.getvalue().splitlines(), [
        'u1,i2,0.9',
        'u2,i1,0.8',
        'u2,i2,0.7',
    ])
```

- [ ] **Step 4: Run the test and confirm the helpers are missing**

Run: `python3 -m unittest tests.test_predict -v`

Expected: FAIL with missing `mask_history_scores`, `build_original_id_lookup`, and `write_prediction_rows` attributes.

- [ ] **Step 5: Commit the failing prediction test**

```bash
git add tests/test_predict.py
git commit -m "test: define streaming prediction behavior"
```

### Task 4: Add reusable cached scoring to `UniSRec`

**Files:**
- Modify: `unisrec.py:196-209`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Extract normalized all-item embedding construction**

Add this method without changing training paths:

```python
def get_full_sort_item_embeddings(self):
    item_embeddings = self.moe_adaptor(self.plm_embedding.weight)
    if self.train_stage == 'transductive_ft':
        item_embeddings = item_embeddings + self.item_embedding.weight
    return F.normalize(item_embeddings, dim=-1)
```

- [ ] **Step 2: Add batch scoring against a supplied cache**

```python
def full_sort_predict_with_item_embeddings(self, interaction, item_embeddings):
    item_seq = interaction[self.ITEM_SEQ]
    item_seq_len = interaction[self.ITEM_SEQ_LEN]
    item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
    seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
    seq_output = F.normalize(seq_output, dim=-1)
    return torch.matmul(seq_output, item_embeddings.transpose(0, 1))
```

- [ ] **Step 3: Preserve the existing RecBole method through delegation**

Replace the duplicated body of `full_sort_predict()` with:

```python
def full_sort_predict(self, interaction):
    item_embeddings = self.get_full_sort_item_embeddings()
    return self.full_sort_predict_with_item_embeddings(interaction, item_embeddings)
```

This deliberately preserves RecBole evaluation semantics while allowing `predict.py` to build the cache once per job.

- [ ] **Step 4: Run syntax validation**

Run: `python3 -m py_compile unisrec.py`

Expected: command exits successfully. Runtime tensor validation remains part of Task 7 because the local environment has no PyTorch or RecBole.

- [ ] **Step 5: Commit the model API change**

```bash
git add unisrec.py
git commit -m "perf: cache item embeddings during prediction"
```

### Task 5: Rewrite `predict.py` as a streaming inference job

**Files:**
- Modify: `predict.py:1-231`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Replace imports and implement the tested helpers**

Keep only `argparse`, `csv`, `json`, `math`, `os`, PyTorch, RecBole, `UniSRec`, and `UniSRecDataset`. Implement:

```python
def mask_history_scores(scores, item_sequences):
    return scores.scatter_(1, item_sequences, float('-inf'))


def build_original_id_lookup(id_tokens, original_ids):
    return [None if str(token) == '[PAD]' else original_ids[str(token)]
            for token in id_tokens]


def write_prediction_rows(writer, user_ids, item_ids, scores,
                          user_lookup, item_lookup):
    row_count = 0
    for user_id, recommended_items, recommended_scores in zip(
            user_ids, item_ids, scores):
        original_user_id = user_lookup[int(user_id)]
        for item_id, score in zip(recommended_items, recommended_scores):
            score = float(score)
            if not math.isfinite(score):
                continue
            writer.writerow((original_user_id, item_lookup[int(item_id)], score))
            row_count += 1
    return row_count
```

- [ ] **Step 2: Implement model loading and one-time lookup/cache construction**

Build `Config` with the existing YAML files and benchmark filenames, use only `config['device']`, load the checkpoint strictly, enter evaluation mode, and construct lookup arrays from `dataset_.field2id_token` plus the two JSON mapping files:

```python
user_lookup = build_original_id_lookup(
    dataset_.field2id_token[config['USER_ID_FIELD']], index2user)
item_lookup = build_original_id_lookup(
    dataset_.field2id_token[config['ITEM_ID_FIELD']], index2item)

with torch.inference_mode():
    item_embeddings = model.get_full_sort_item_embeddings()
```

- [ ] **Step 3: Stream each predicted batch directly to the sole CSV**

Open `<result_save_path>/<dataset>-recommendations.csv` with `newline=''` and UTF-8, write the header once, and inside `torch.inference_mode()`:

```python
interaction = batched_data[0].to(config['device'])
item_sequences = interaction[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']]
scores = model.full_sort_predict_with_item_embeddings(interaction, item_embeddings)
mask_history_scores(scores, item_sequences)
k = min(top_k, max(tot_item_num - 1, 0))
if k == 0:
    continue
topk_scores, topk_items = torch.topk(scores, k=k, dim=1)
row_count += write_prediction_rows(
    writer,
    interaction[config['USER_ID_FIELD']].detach().cpu().tolist(),
    topk_items.detach().cpu().tolist(),
    topk_scores.detach().cpu().tolist(),
    user_lookup,
    item_lookup,
)
```

Print the result path and total row count after the file closes. Do not create another result file or retain batch outputs.

- [ ] **Step 4: Reduce the command-line parser to pure inference parameters**

Expose only:

```python
parser.add_argument('-d', required=True, help='dataset name')
parser.add_argument('-fp', required=True, help='fine-tuned model path')
parser.add_argument('-t', type=int, default=50, help='number of recommendations per user')
parser.add_argument('-sp', default='outputs/results/', help='result directory')
parser.add_argument('--data-path', help='parent directory of the dataset')
```

Retain the positive TopK validation and call `predictor()` with these five values.

- [ ] **Step 5: Run helper tests and syntax validation**

Run: `python3 -m unittest tests.test_predict -v && python3 -m py_compile predict.py unisrec.py`

Expected: all prediction helper tests pass and both files compile.

- [ ] **Step 6: Commit the prediction rewrite**

```bash
git add predict.py tests/test_predict.py
git commit -m "refactor: stream pure model recommendations"
```

### Task 6: Align README with the executable interface

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update structure, quick-start, and data-flow sections**

Describe `predict.py` as pure model inference. Remove every prediction example or flow edge involving item metadata, eligible-item inputs, ODPS output, details CSV, category diversification, 800 candidates, and the 150000-user limit. The full-pipeline example becomes:

```bash
bash run.sh --stage all \
  --dataset catalog \
  --input-csv /path/to/interactions.csv \
  --plm-path /path/to/text-encoder \
  --max-seq-length 50 \
  --top-k 50 \
  --work-dir /path/to/work-dir \
  --python python3
```

The prediction edge in the Mermaid graph must end only at:

```text
<work-dir>/results/<dataset>-recommendations.csv
```

- [ ] **Step 2: Replace the standalone prediction section and parameter table**

Document this runnable command:

```bash
bash run.sh --stage predict \
  --dataset catalog \
  --finetuned-checkpoint /path/to/UniSRec-catalog-finetuned.pth \
  --top-k 50 \
  --work-dir /path/to/work-dir \
  --python python3
```

List exactly the six prediction-stage Bash parameters approved by the design. Explain that all prediction users are processed, padding and their entire input history are excluded, fewer than TopK rows can be produced when too few unseen items remain, and rows are ordered by descending model score for each user.

- [ ] **Step 3: Confirm removed concepts are absent and current concepts remain**

Run:

```bash
rg -n 'item-metadata|eligible-items|output-table|details-|150000|800 个候选|品类打散' README.md
rg -n -- '--stage predict|--finetuned-checkpoint|--top-k|recommendations.csv|user_id,item_id,score' README.md
```

Expected: the first command produces no output; the second finds the standalone command, parameter documentation, data flow, and result schema.

- [ ] **Step 4: Commit the documentation**

```bash
git add README.md
git commit -m "docs: document pure inference prediction"
```

### Task 7: Final verification and handoff

**Files:**
- Verify: `predict.py`
- Verify: `unisrec.py`
- Verify: `run.sh`
- Verify: `tests/test_predict.py`
- Verify: `tests/test_run.py`
- Verify: `README.md`

- [ ] **Step 1: Run the focused tests and syntax checks**

Run:

```bash
python3 -m unittest tests.test_predict tests.test_run -v
bash -n run.sh
python3 -m py_compile predict.py unisrec.py
```

Expected: all tests pass and syntax checks exit successfully.

- [ ] **Step 2: Run the complete local test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests that do not require production-only dependencies pass. If a dependency-specific test is skipped or cannot import, record the exact test and error rather than claiming it passed.

- [ ] **Step 3: Inspect scope and whitespace**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- predict.py unisrec.py run.sh tests/test_predict.py tests/test_run.py README.md
```

Expected: no whitespace errors; only scoped implementation files plus the user's pre-existing YAML modifications appear. The YAML files remain unstaged and unchanged by this work.

- [ ] **Step 4: Run a production-environment smoke test when the GPU Pod is available**

Run the standalone prediction stage with a real fine-tuned checkpoint and a small test dataset, then verify:

```bash
head -5 <work-dir>/results/<dataset>-recommendations.csv
```

Expected: the first row is `user_id,item_id,score`; subsequent item IDs are absent from each corresponding input sequence; logs show one prediction start, one output path, and a positive row count. This check validates real PyTorch, RecBole, CUDA, checkpoint, and dataset compatibility that the local environment cannot execute.

- [ ] **Step 5: Commit any final scoped corrections**

If verification required corrections, stage only the files in this plan and commit them with a message describing the correction. Do not stage `configs/UniSRec.yaml`, `configs/finetune.yaml`, or `configs/pretrain.yaml`.
