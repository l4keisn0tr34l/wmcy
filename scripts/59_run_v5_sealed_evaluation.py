#!/usr/bin/env python3
"""Consume one permanent V5 test-access claim, export, evaluate, preserve evidence."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_model_protocol import file_sha256
from src.cyberwm.v5_evaluation_seal import FREEZE, OUTPUT, claim_attempt, durable_json, verify_freeze


def main():
    # All dependencies, scaler, checkpoints, validation lineage and runtime
    # checked BEFORE the permanent claim and BEFORE any test content is read.
    freeze = verify_freeze()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing test access')
    claim = claim_attempt(OUTPUT, file_sha256(FREEZE))
    work = claim / 'work'; work.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    try:
        with (work / 'execution.log').open('x') as log:
            for mode in ('passive', 'action', 'passive_action'):
                subprocess.run([sys.executable, str(ROOT / 'scripts/47_build_v5_sequences.py'),
                    '--split', 'test', '--mode', mode, '--unlock-test', '--out-dir', str(work / 'sequences')],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Hash and structurally check every bundle in the evaluator before inference.
            subprocess.run([sys.executable, str(ROOT / 'scripts/57_evaluate_v5_models.py'),
                '--split', 'test', '--sequences-dir', str(work / 'sequences'),
                '--evaluation-freeze', str(FREEZE), '--out', str(work / 'report.json')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((work / 'report.json').read_text())
        if report['status'] != 'SEALED_TEST_COMPLETE': raise ValueError('incomplete report')
        files = {str(p.relative_to(work)): file_sha256(p) for p in sorted(work.rglob('*')) if p.is_file()}
        durable_json(work / 'provenance.json', {
            'status': 'SEALED_TEST_COMPLETE', 'started_utc': started,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_freeze_sha256': file_sha256(FREEZE),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'artifact_sha256_before_provenance': files, 'rerun_policy': 'forbidden; permanent claim remains'})
        os.rename(work, OUTPUT / 'sealed_test')
        durable_json(claim / 'complete.json', {'status': 'SEALED_TEST_COMPLETE'})
    except BaseException as exc:
        # No deletion or automatic retry, even if export failed before inference.
        durable_json(claim / 'failed.json', {'error': repr(exc), 'evidence': str(work), 'retry': False})
        raise
    print(f"V5 sealed evaluation complete -> {OUTPUT / 'sealed_test'}")
    return 0


if __name__ == '__main__': raise SystemExit(main())
