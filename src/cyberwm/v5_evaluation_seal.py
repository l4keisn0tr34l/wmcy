"""Evaluation-only integrity and persistent single-attempt controls (not training)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

from src.cyberwm.v5_model_protocol import file_sha256

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'configs/mvp_v5_evaluation_freeze.json'
OUTPUT = ROOT / 'outputs/mvp_v5'
CLAIM = OUTPUT / 'sealed_test_attempt'
REGIMES = ('scratch', 'friday', 'v4')
SCALER = 'outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz'


def runtime_versions():
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda)


def check_hashes(mapping, root=ROOT):
    for relative, digest in mapping.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f'frozen artifact changed/missing: {relative}')


def no_previous_attempt(output=OUTPUT):
    paths = [output / 'sealed_test', output / 'sealed_test_attempt']
    paths += list(output.glob('_failed_sealed_test*')) + list(output.glob('.v5-sealed-test-*'))
    paths += [output / 'sequences' / mode / 'test' for mode in ('action', 'passive', 'passive_action')]
    if any(p.exists() for p in paths):
        raise PermissionError('V5 test already accessed/claimed or interrupted; automatic retry forbidden')


def durable_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def claim_attempt(output, freeze_sha256):
    no_previous_attempt(output)
    claim = output / 'sealed_test_attempt'
    claim.mkdir()  # Exclusive, persistent and never removed, even on failure.
    fd = os.open(output, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    durable_json(claim / 'claim.json', {'freeze_sha256': freeze_sha256, 'pid': os.getpid()})
    return claim


def claim_inference(claim, freeze_sha256, sequences, report):
    record = json.loads((claim / 'claim.json').read_text())
    if record['freeze_sha256'] != freeze_sha256:
        raise PermissionError('run claim belongs to another freeze')
    if record['pid'] != os.getppid() or (claim / 'failed.json').exists() or (claim / 'complete.json').exists():
        raise PermissionError('only the live claiming wrapper may start its evaluator')
    if sequences.resolve() != (claim / 'work/sequences').resolve() or report.resolve() != (claim / 'work/report.json').resolve():
        raise PermissionError('test evaluator must use the claimed staging paths')
    durable_json(claim / 'inference_started.json', {'freeze_sha256': freeze_sha256})


def checkpoint_contract(root=ROOT):
    """Verify original selected checkpoints against reports; no test I/O."""
    audit = json.loads((root / 'outputs/mvp_v5/validation_audit/audit.json').read_text())
    hashes = audit['checkpoint_sha256']; thresholds = {}; selections = {}; artifacts = {}
    for track, directory, stem in [('action', 'action_model', 'action_graph_rssm'),
                                    ('passive', 'passive_branch', 'passive_branch')]:
        path = root / f'outputs/mvp_v5/{directory}/validation_report.json'
        report = json.loads(path.read_text()); thresholds[track] = {}; selections[track] = {}
        artifacts[str(path.relative_to(root))] = file_sha256(path)
        results = report['results']
        primary = min(results, key=lambda n: (round(results[n]['selected_validation_selection'], 8), n))
        if primary != report['primary_initialization_selected_before_test'] or primary != audit[f'{track}_primary']:
            raise ValueError('primary selection lineage mismatch')
        for regime in REGIMES:
            path = root / f'models/mvp_v5_{stem}_{regime}.pt'
            check_hashes({str(path.relative_to(root)): hashes[track][regime]}, root)
            c = torch.load(path, map_location='cpu', weights_only=False)
            r = results[regime]
            if c['validation'] != r or c['v5_test_loaded'] is not False:
                raise ValueError('checkpoint/report validation mismatch')
            if c['scaler_sha256'] != audit['shared_scaler_sha256']:
                raise ValueError('checkpoint scaler lineage mismatch')
            score_key = 'validation_selection' if track == 'action' else 'best_validation_selection'
            selected = min(r['candidates'], key=lambda x: (round(x[score_key], 8), x['seed']))
            if r['selected_seed'] != selected['seed']:
                raise ValueError('seed selection mismatch')
            structural = selected['trained_cpu_audit']
            delta = (structural['all_120_cpu_equivariance_max_delta'] if track == 'action'
                     else max(structural['all_120_cpu_equivariance'].values()))
            if structural['future_perturbation_max_delta'] != 0 or delta >= 1e-5:
                raise ValueError('selected checkpoint structural gate failed')
            metrics = r['selected_validation_metrics']
            if track == 'action':
                threshold = metrics['lm_threshold_from_validation']
                if metrics['same_context_counterfactual']['mean_permit_minus_block_lm_probability'] <= 0:
                    raise ValueError('action direction gate')
            else:
                threshold = metrics['exact_lm']['validation_best_threshold']
                if metrics['future_graph']['mean_pairwise_branch_mae'] <= 0:
                    raise ValueError('branch diversity gate')
            if not np.isfinite(threshold) or not 0 <= threshold <= 1: raise ValueError('invalid threshold')
            thresholds[track][regime] = threshold
            selections[track][regime] = {'seed': r['selected_seed'], 'epoch': r.get('selected_epoch', 400)}
            artifacts[str(path.relative_to(root))] = hashes[track][regime]
    return hashes, thresholds, selections, artifacts


def verify_freeze(path=FREEZE, root=ROOT, committed=True):
    f = json.loads(path.read_text())
    if f.get('status') != 'FROZEN_FOR_ONE_SEALED_TEST' or f.get('test_unlock') is not True or f.get('allowed_runs') != 1:
        raise PermissionError('evaluation is not frozen for one run')
    check_hashes(f['source_sha256'], root); check_hashes(f['artifact_sha256'], root)
    if f['runtime_versions'] != runtime_versions(): raise ValueError('runtime differs from freeze')
    if committed:
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
        if status.strip(): raise PermissionError('commit reviewed freeze before test access')
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + str(path.relative_to(root))], cwd=root)
        if blob != path.read_bytes(): raise PermissionError('freeze differs from committed blob')
    return f
