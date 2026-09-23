"""Embed the proven Phase 4 feature builder and the single Phase 4.1 experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE4 = ROOT.parent / "phase4_final" / "phase4_blend.py"


def cell(kind: str, source: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def main() -> None:
    phase4 = PHASE4.read_text(encoding="utf-8")
    phase41 = (ROOT / "phase41_ensemble.py").read_text(encoding="utf-8")
    cells = [
        cell("markdown", """# Phase 4.1 — one CPU-only cache ensemble (private)

Attach: (1) official competition data, (2) saved **Phase 2 private** output (including its retrieval and per-engine rerank caches), (3) saved **Phase 4 fixed** output (including `phase4_report.json`, `artifacts_phase4`, `submission_phase4_private_final.json` and config), (4) Phase 3 delta bundle (only for its offline project wheel). Internet Off; **CPU** accelerator. No models, GPU, retrieval, prepare, or rerank jobs are executed.

If and only if the ensemble beats *reproduced Phase 4 OOF* overall with no declining inner fold and exactly reconstructs the saved Phase 4 private answer sets, a single Phase 4.1 zip is created. Otherwise only a report is written; retain your Phase 4 submission. Saved Phase 4 outputs are never modified.
"""),
        cell("code", """import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

INPUT_ROOT = Path('/kaggle/input')
WORK = Path('/kaggle/working/legalir-phase41-private')
RUNTIME = Path('/kaggle/working/legalir-phase41-runtime')
WORK.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

def exactly_one(items, label):
    found = list(items)
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one {label}, found {len(found)}: {found}')
    return found[0]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

REPORT_FILE = exactly_one(
    (p for p in INPUT_ROOT.rglob('phase4_report.json')
     if read(p).get('experiment_id') == 'phase4-phase2-legal-bge-protected-blender'
     and (p.parent / 'submission_phase4_private_final.json').is_file()),
    'saved Phase 4 output report',
)
PHASE4_OUTPUT = REPORT_FILE.parent
REPORT = read(REPORT_FILE)
private_matches = [
    p for p in INPUT_ROOT.rglob('private-official.json')
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == REPORT['test_sha256']
]
official_matches = [
    p for p in private_matches
    if (p.parent / 'train.json').is_file() and any(p.parent.rglob('context_*.json'))
]
PRIVATE_FILE = exactly_one(official_matches or private_matches, 'official private input matching Phase 4 report')
STATE_FILE = exactly_one(
    (p for p in INPUT_ROOT.rglob('inference_input_state.json')
     if (lambda s: s.get('filename') == 'private-official.json'
         and s.get('sha256') == REPORT['test_sha256']
         and s.get('questions') == REPORT['test_questions']
         and s.get('project_commit') == REPORT['project_commit'])(read(p))
     and (p.parent / 'artifacts_phase2_harrier').is_dir()),
    'matching saved Phase 2 private output',
)
PHASE2_ARTIFACTS = STATE_FILE.parent / 'artifacts_phase2_harrier'
DELTA_MANIFEST = exactly_one(
    (p for p in INPUT_ROOT.rglob('bundle_manifest.json')
     if read(p).get('experiment_id') == 'phase3-rerankers-harrier-retrieval'
     and read(p).get('project_commit') == REPORT['project_commit']),
    'Phase 3 delta bundle with matching project commit',
)
DELTA_ROOT = DELTA_MANIFEST.parents[1]
WHEEL = exactly_one((DELTA_ROOT / 'wheels').glob('uit_legalir-*.whl'), 'offline project wheel')
print('Phase 4 output:', PHASE4_OUTPUT)
print('Phase 2 caches:', PHASE2_ARTIFACTS)
print('Private input:', PRIVATE_FILE)
print('Project wheel:', WHEEL)
"""),
        cell("code", """# Only install the pinned project code; Kaggle's built-in NumPy, sklearn, PyYAML
# are used as-is. This step cannot fetch dependencies or models from the network.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--target', str(RUNTIME), str(WHEEL)], check=True)
runtime_env = os.environ.copy()
runtime_env['PYTHONPATH'] = str(RUNTIME) + os.pathsep + str(WORK)
runtime_env['PYTHONNOUSERSITE'] = '1'
subprocess.run([sys.executable, '-c',
                'import legalir, sklearn, numpy, yaml; print("CPU dependencies OK", sklearn.__version__)'],
               check=True, env=runtime_env)
"""),
        cell("code", """# The exact Phase 4 feature code used by the completed fixed notebook.
(WORK / 'phase4_blend.py').write_text(""" + repr(phase4) + """, encoding='utf-8')
(WORK / 'phase41_ensemble.py').write_text(""" + repr(phase41) + """, encoding='utf-8')
subprocess.run([sys.executable, str(WORK / 'phase41_ensemble.py'), str(PHASE4_OUTPUT),
                str(PHASE2_ARTIFACTS), str(WORK), str(PRIVATE_FILE)],
               check=True, env=runtime_env, cwd=WORK)
result = read(WORK / 'phase41_report.json')
if result['approved']:
    print('ELIGIBLE SINGLE SUBMISSION:', result['zip'])
else:
    print('DO NOT SPEND LAST SUBMISSION: retain Phase 4.', result['reason'])
"""),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "legalir_phase41_cache_only.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()