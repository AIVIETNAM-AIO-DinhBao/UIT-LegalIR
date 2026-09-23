from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    blender = (ROOT / "phase4_blend.py").read_text(encoding="utf-8")
    preflight_source = """import sys
from pathlib import Path
import yaml
from legalir.rerank import PairwiseReranker
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
engine = PairwiseReranker(config, 'legal_reranker')
order = engine.rank('điều kiện cấp giấy phép', ['văn bản có quy định cấp giấy phép', 'văn bản không liên quan'])
assert sorted(order) == [0, 1]
engine.close()
print('Phase 4 legal reranker preflight passed')
"""
    cells = [
        markdown(
            """# Phase 4 Final — Phase 2 top-80 + Vietnamese Legal BGE + protected rank blender

Attach the competition data, Phase 2 Harrier bundle, Phase 3 reranker delta bundle, and the saved output of the completed Phase 2 private notebook. Select RTX Pro 6000 and set Internet to Off.

The notebook reuses every expensive Phase 2 retrieval/Jina/Vietnamese-reranker artifact. It runs only the small Vietnamese legal BGE reranker over the existing top-80 candidates, then trains a cross-fitted CPU rank blender with protected promotion from the proven Phase 2 top five. Previous submissions are never modified.
"""
        ),
        code(
            """import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from pathlib import Path
EXPERIMENT_ID = 'phase4-phase2-legal-bge-protected-blender'
DATASET_DIR = Path('/kaggle/input/datasets/tonioz/uit-dsc-task1')
TEST_FILENAME = 'private-official.json'
if TEST_FILENAME not in {'private-official.json', 'public-official.json'}:
    raise ValueError(TEST_FILENAME)
TEST_LABEL = 'private' if TEST_FILENAME == 'private-official.json' else 'public'
PHASE2_BUNDLE = Path('/kaggle/input/datasets/boinhbo/legalir-phase2-harrier-bundle/legalir-phase2-harrier-bundle')
PHASE3_DELTA = Path('/kaggle/input/datasets/boinhbo/legalir-phase3-reranker-delta/legalir-phase3-reranker-delta')
PHASE2_ARTIFACTS_OVERRIDE = None  # Set Path(...) only if auto-discovery finds multiple complete copies.
WORK_DIR = Path(f'/kaggle/working/legalir-phase4-{TEST_LABEL}-final')
RUNTIME_DIR = Path('/kaggle/working/legalir-phase4-runtime')
"""
        ),
        code(
            """import json
import shutil
import subprocess
import sys
import time

def run(*command, cwd=None, env=None):
    print('+', ' '.join(map(str, command)))
    started = time.perf_counter()
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)
    print(f'Completed in {(time.perf_counter() - started) / 60:.1f} minutes')

phase2_manifest = json.loads((PHASE2_BUNDLE / 'manifests' / 'bundle_manifest.json').read_text(encoding='utf-8'))
delta_manifest = json.loads((PHASE3_DELTA / 'manifests' / 'bundle_manifest.json').read_text(encoding='utf-8'))
if phase2_manifest.get('experiment_id') != 'phase2-vietlegal-harrier-0.6b':
    raise RuntimeError('Wrong Phase 2 bundle')
if delta_manifest.get('experiment_id') != 'phase3-rerankers-harrier-retrieval':
    raise RuntimeError('Wrong Phase 3 delta bundle')
if 'legal_reranker' not in {row['name'] for row in delta_manifest['models']}:
    raise RuntimeError('Phase 3 delta lacks legal_reranker')
for manifest, bundle in ((phase2_manifest, PHASE2_BUNDLE), (delta_manifest, PHASE3_DELTA)):
    for record in manifest['files']:
        path = bundle / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise RuntimeError(f'Missing or truncated bundle file: {path}')

WORK_DIR.mkdir(parents=True, exist_ok=True)
if RUNTIME_DIR.exists():
    shutil.rmtree(RUNTIME_DIR)
RUNTIME_DIR.mkdir(parents=True)
run(sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--ignore-installed', '--target', RUNTIME_DIR, '--find-links', PHASE3_DELTA / 'wheels', '-r', PHASE3_DELTA / 'requirements-offline.txt')
project_wheels = sorted((PHASE3_DELTA / 'wheels').glob('uit_legalir-*.whl'))
if len(project_wheels) != 1:
    raise RuntimeError(f'Expected one project wheel, found {project_wheels}')
run(sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--ignore-installed', '--target', RUNTIME_DIR, project_wheels[0])
runtime_env = os.environ.copy()
runtime_env['PYTHONPATH'] = str(RUNTIME_DIR)
runtime_env['PYTHONNOUSERSITE'] = '1'
probe = "import sklearn, torch, transformers; assert torch.cuda.is_available(); print('sklearn', sklearn.__version__); print('GPU', torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
run(sys.executable, '-c', probe, env=runtime_env)
print('Runtime project commit:', delta_manifest['project_commit'])
"""
        ),
        code(
            """import hashlib
contexts_source = DATASET_DIR / 'selected-contexts' / 'selected-contexts'
if not contexts_source.is_dir():
    raise FileNotFoundError(f'Missing corpus: {contexts_source}')
for filename in ('train.json', TEST_FILENAME):
    if not (DATASET_DIR / filename).is_file():
        raise FileNotFoundError(DATASET_DIR / filename)

def ensure_link(destination, source, is_directory=False):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f'Refusing to overwrite: {destination}')
    destination.symlink_to(source, target_is_directory=is_directory)

ensure_link(WORK_DIR / 'selected-contexts', contexts_source, True)
for filename in ('train.json', TEST_FILENAME):
    ensure_link(WORK_DIR / filename, DATASET_DIR / filename)
test_bytes = (DATASET_DIR / TEST_FILENAME).read_bytes()
test_sha256 = hashlib.sha256(test_bytes).hexdigest()
test_question_count = len(json.loads(test_bytes))
print('Corpus documents:', sum(1 for _ in contexts_source.glob('context_*.json')))
print('Test questions:', test_question_count, 'sha256:', test_sha256)

required_phase2 = {
    'prepare_manifest.json', 'corpus.jsonl', 'chunks_short.jsonl', 'chunks_long.jsonl',
    'lexical_short.pkl', 'train_questions.jsonl', 'public_questions.jsonl',
    'first_stage_weights.json', 'final_weights.json', 'retrieval_train.json',
    'retrieval_public.json', 'rerank_train_0.json', 'rerank_public.json',
}
def complete_phase2(path):
    return path.is_dir() and all((path / name).is_file() for name in required_phase2) and all(
        all((path / 'dense' / model / name).is_file() for name in ('vectors.npy', 'index.faiss', 'chunks.json'))
        for model in ('vietlegal_harrier', 'vietnamese_embedding', 'nemotron')
    ) and all(
        all((path / 'question_memory' / model / name).is_file() for name in ('vectors.npy', 'questions.json'))
        for model in ('vietlegal_harrier', 'vietnamese_embedding', 'nemotron')
    )

if PHASE2_ARTIFACTS_OVERRIDE is not None:
    PHASE2_ARTIFACTS = Path(PHASE2_ARTIFACTS_OVERRIDE)
    if not complete_phase2(PHASE2_ARTIFACTS):
        raise RuntimeError(f'Incomplete Phase 2 artifacts: {PHASE2_ARTIFACTS}')
else:
    candidates = [path for path in Path('/kaggle/input').rglob('artifacts_phase2_harrier') if complete_phase2(path)]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one complete Phase 2 artifact directory, found {candidates}. Set PHASE2_ARTIFACTS_OVERRIDE.')
    PHASE2_ARTIFACTS = candidates[0]

state_path = PHASE2_ARTIFACTS.parent / 'inference_input_state.json'
if not state_path.is_file():
    raise RuntimeError(f'Phase 2 output lacks inference input state: {state_path}')
state = json.loads(state_path.read_text(encoding='utf-8'))
if state.get('filename') != TEST_FILENAME or state.get('sha256') != test_sha256 or state.get('questions') != test_question_count:
    raise RuntimeError(f'Phase 2 private cache fingerprint mismatch: {state}')
print('Using complete matching Phase 2 artifacts:', PHASE2_ARTIFACTS)
"""
        ),
        code(
            """import yaml
config = yaml.safe_load((PHASE2_BUNDLE / 'configs' / 'kaggle_rtx_pro_6000.yaml').read_text(encoding='utf-8'))
config['paths']['public_file'] = TEST_FILENAME
config['paths']['artifacts_dir'] = 'artifacts_phase4'
for name in ('vietlegal_harrier', 'vietnamese_embedding', 'nemotron', 'jina', 'vietnamese_reranker'):
    config['models'][name]['local_path'] = str(PHASE2_BUNDLE / 'models' / name)
    config['models'][name]['local_files_only'] = True
phase3_config = yaml.safe_load((PHASE3_DELTA / 'configs' / 'kaggle_rtx_pro_6000.yaml').read_text(encoding='utf-8'))
config['models']['legal_reranker'] = phase3_config['models']['legal_reranker']
config['models']['legal_reranker']['local_path'] = str(PHASE3_DELTA / 'models' / 'legal_reranker')
config['models']['legal_reranker']['local_files_only'] = True
config['reranking']['rerank_top_k'] = 80
config['reranking']['pairwise_max_length'] = 512
config['reranking']['pairwise_evidence_tokens'] = 440
config['reranking']['pairwise_batch_size'] = 64
config['reranking']['query_batch_size'] = 16
config['reranking']['checkpoint_every_questions'] = 32
config['validation']['reranker_tuning_folds'] = [0]

artifacts = WORK_DIR / config['paths']['artifacts_dir']
artifacts.mkdir(parents=True, exist_ok=True)
for source in PHASE2_ARTIFACTS.iterdir():
    if source.name in {'model_manifest.json', 'final_weights.json'}:
        continue
    if source.name.startswith('rerank_train_0_legal_reranker') or source.name.startswith('rerank_public_legal_reranker'):
        continue
    destination = artifacts / source.name
    if not destination.exists() and not destination.is_symlink():
        destination.symlink_to(source, target_is_directory=source.is_dir())
shutil.copy2(PHASE2_ARTIFACTS / 'final_weights.json', artifacts / 'phase2_final_weights.json')
config_path = WORK_DIR / 'kaggle_rtx_pro_6000_phase4.yaml'
config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
print(config_path.read_text(encoding='utf-8'))
"""
        ),
        code(
            "# Only the newly added model needs a preflight; Phase 2 models already produced the attached caches.\n"
            + "preflight = WORK_DIR / 'phase4_legal_preflight.py'\n"
            + "preflight.write_text("
            + repr(preflight_source)
            + ", encoding='utf-8')\n"
            + "run(sys.executable, preflight, config_path, cwd=WORK_DIR, env=runtime_env)\n\n"
            + "base = [sys.executable, '-m', 'legalir']\n"
            + "def legalir(*args):\n    run(*base, *args, cwd=WORK_DIR, env=runtime_env)\n\n"
            + "# Audit loads/counts all deployed checkpoints but does not recompute retrieval or Phase 2 reranking.\n"
            + "legalir('audit', '--config', config_path)\n"
            + "legalir('rerank', '--config', config_path, '--split', 'train', '--fold', '0', '--engine', 'legal_reranker', '--resume')\n"
            + "legalir('rerank', '--config', config_path, '--split', 'public', '--engine', 'legal_reranker', '--resume')\n"
        ),
        code(
            "blender_script = WORK_DIR / 'phase4_blend.py'\n"
            + "blender_script.write_text("
            + repr(blender)
            + ", encoding='utf-8')\n"
            + "run(sys.executable, blender_script, WORK_DIR, config_path, cwd=WORK_DIR, env=runtime_env)\n"
            + "submission_json = WORK_DIR / 'submission_phase4_private_final.json'\n"
            + "submission_zip = WORK_DIR / 'submission_phase4_private_final.zip'\n"
            + "run('zip', '-j', submission_zip, submission_json)\n"
            + "print('PHASE 4 FINAL:', submission_zip)\n"
        ),
        code(
            """report = json.loads((WORK_DIR / 'phase4_report.json').read_text(encoding='utf-8'))
model_manifest = json.loads((artifacts / 'model_manifest.json').read_text(encoding='utf-8'))
submission = json.loads((WORK_DIR / 'submission_phase4_private_final.json').read_text(encoding='utf-8'))
phase2_submission_path = PHASE2_ARTIFACTS.parent / 'submission_phase2_private_harrier.json'
comparison = None
if phase2_submission_path.is_file():
    phase2_submission = json.loads(phase2_submission_path.read_text(encoding='utf-8'))
    shared = set(submission).intersection(phase2_submission)
    comparison = {
        'questions_compared': len(shared),
        'changed_answer_sets': sum(set(submission[q]['answer']) != set(phase2_submission[q]['answer']) for q in shared),
        'changed_top1': sum(submission[q]['answer'][0] != phase2_submission[q]['answer'][0] for q in shared),
    }
report.update({
    'test_file': TEST_FILENAME,
    'test_questions': test_question_count,
    'test_sha256': test_sha256,
    'project_commit': delta_manifest['project_commit'],
    'models': model_manifest['models'],
    'total_parameters': model_manifest['total_parameters'],
    'private_comparison_to_phase2': comparison,
    'submission_zip': str(WORK_DIR / 'submission_phase4_private_final.zip'),
})
(WORK_DIR / 'phase4_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "legalir_phase4_final.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()