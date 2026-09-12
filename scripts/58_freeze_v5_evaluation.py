#!/usr/bin/env python3
"""Freeze fully verified V5 lineage before one permanent test-access claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_model_protocol import file_sha256
from src.cyberwm.v5_evaluation_seal import (SCALER, checkpoint_contract, check_hashes,
    durable_json, no_previous_attempt, runtime_versions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args(); no_previous_attempt()
    path = ROOT / 'configs/mvp_v5_evaluation_freeze.json'
    training = json.loads((ROOT / 'configs/mvp_v5_training_freeze.json').read_text())
    check_hashes(training['source_sha256'])
    check_hashes({'outputs/mvp_v5/sequences/' + k: v for k, v in training['export_artifact_sha256'].items()})
    check_hashes({SCALER: training['scaler_sha256']})
    audit_path = 'outputs/mvp_v5/validation_audit/audit.json'
    audit = json.loads((ROOT / audit_path).read_text())
    if audit['training_freeze_sha256'] != file_sha256(ROOT / 'configs/mvp_v5_training_freeze.json'):
        raise ValueError('validation lineage changed')
    if audit['status'] != 'PASS_VALIDATION_ONLY_TEST_SEALED' or not audit['test_exports_absent']:
        raise ValueError('validation audit failed')
    hashes, thresholds, selections, artifacts = checkpoint_contract()
    if audit['shared_scaler_sha256'] != training['scaler_sha256']: raise ValueError('scaler lineage mismatch')
    artifacts.update({audit_path: file_sha256(ROOT / audit_path), SCALER: training['scaler_sha256']})
    artifacts.update({'outputs/mvp_v5/sequences/' + k: v for k, v in training['export_artifact_sha256'].items()})
    smoke_path = 'outputs/mvp_v5/evaluation_protocol/reviewed_validation_smoke/report.json'
    smoke = json.loads((ROOT / smoke_path).read_text())
    if smoke['test_access'] or smoke['status'] != 'VALIDATION_EVALUATOR_SMOKE_COMPLETE': raise ValueError('smoke status')
    if smoke['evaluator_sha256'] != file_sha256(ROOT / 'scripts/57_evaluate_v5_models.py'):
        raise ValueError('evaluator changed since reviewed smoke')
    if smoke['thresholds'] != thresholds or smoke['runtime_versions'] != runtime_versions(): raise ValueError('smoke contract drift')
    artifacts[smoke_path] = file_sha256(ROOT / smoke_path)
    predictions = str(Path(smoke_path).with_suffix('.predictions.npz'))
    check_hashes({predictions: smoke['predictions_sha256']})
    artifacts[predictions] = smoke['predictions_sha256']
    # Include all local Python modules and all plan CSVs consulted by validator46.
    sources = list(ROOT.glob('src/cyberwm/*.py')) + list(ROOT.glob('configs/*.csv'))
    sources += [ROOT / p for p in (
        'configs/mvp_v5_capture_freeze.json', 'configs/mvp_v5_training_freeze.json',
        'configs/mvp_v5_training_protocol.json', 'scripts/15_train_rssm.py',
        'scripts/46_validate_v5_plan.py', 'scripts/47_build_v5_sequences.py',
        'scripts/56_audit_v5_validation_models.py', 'scripts/57_evaluate_v5_models.py',
        'scripts/58_freeze_v5_evaluation.py', 'scripts/59_run_v5_sealed_evaluation.py',
        'scripts/60_test_v5_evaluation_seal.py', 'docs/V5_FINAL_REVIEW.md', 'docs/V5_EVALUATION_PROTOCOL.md')]
    freeze = {
        'status': 'FROZEN_FOR_ONE_SEALED_TEST', 'protocol_id': 'mvp_v5_sealed_evaluation_reviewed_v1',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in sources},
        'artifact_sha256': artifacts, 'runtime_versions': runtime_versions(),
        'checkpoint_sha256': hashes, 'thresholds': thresholds, 'selections': selections,
        'primary_action': audit['action_primary'], 'primary_passive': audit['passive_primary'],
        'shared_scaler_sha256': training['scaler_sha256'],
        'test_expected_samples': {'action': 8, 'passive': 336, 'passive_action': 8},
        'device': 'cuda', 'rollout': 'deterministic prior means; no Monte Carlo calibration claim',
        'test_metrics': ['state MAE overall/group/active/quiet/horizon; persistence', 'edge AP/prevalence',
            'ATT&CK AP/support', 'LM AP/Brier/F1 at fixed0.5 and original checkpoint validation threshold',
            'LM pair AP/top1', 'factual versus opposite action', 'passive expected/oracle/diversity',
            'scenario/profile/episode metrics', 'primary passive exact pre-first-event warning lead and false alerts',
            'intent probe at metadata-chosen pre-SSH cutoff; four matched families'],
        'test_unlock': True, 'allowed_runs': 1, 'post_test_retuning': False,
        'limitations': ['independent small paired families, not packet-identical',
            'one synthetic topology and deterministic chosen-action outcomes',
            'oracle coverage, not deployable accuracy; no calibrated uncertainty',
            'different passive/action training sets prevent a pure conditioning ablation']}
    if args.check_only:
        print(json.dumps({'status': 'PREFREEZE_CHECK_PASS_NO_UNLOCK', 'thresholds': thresholds,
                          'selections': selections, 'source_count': len(sources)}, indent=2)); return 0
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise PermissionError('commit reviewed sources before freeze')
    durable_json(path, freeze)
    print(f'Created {path}; inspect and commit before test access')
    return 0


if __name__ == '__main__': raise SystemExit(main())
