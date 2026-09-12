#!/usr/bin/env python3
"""Synthetic-only negative tests of V5 seal and intent cutoff; never load V5 test."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cyberwm.v5_evaluation_seal import (claim_attempt, claim_inference, check_hashes,
    verify_freeze, runtime_versions, no_previous_attempt)
from src.cyberwm.v5_model_protocol import file_sha256


def script(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class SealTests(unittest.TestCase):
    def test_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'scaler';p.write_bytes(b'original')
            h={'scaler':file_sha256(p)};check_hashes(h,root)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):check_hashes(h,root)
            p.unlink()
            with self.assertRaises(ValueError):check_hashes(h,root)

    def test_claim_blocks_retry_and_parallel_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);claim=claim_attempt(out,'abc')
            with self.assertRaises(PermissionError):claim_attempt(out,'abc')
            (claim/'work').mkdir();(claim/'failed.json').write_text('{}')
            with self.assertRaises(PermissionError):no_previous_attempt(out)

    def test_inference_ticket_paths_freeze_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            claim=claim_attempt(Path(tmp),'abc');sequences=claim/'work/sequences';report=claim/'work/report.json'
            with self.assertRaises(PermissionError):claim_inference(claim,'wrong',sequences,report)
            with self.assertRaises(PermissionError):claim_inference(claim,'abc',sequences,claim/'other.json')
            with self.assertRaises(PermissionError):claim_inference(claim,'abc',sequences,report)
            with patch('src.cyberwm.v5_evaluation_seal.os.getppid',return_value=__import__('os').getpid()):
                claim_inference(claim,'abc',sequences,report)
                with self.assertRaises(FileExistsError):claim_inference(claim,'abc',sequences,report)
                (claim/'failed.json').write_text('{}')
                with self.assertRaises(PermissionError):claim_inference(claim,'abc',sequences,report)

    def test_failed_and_partial_old_run_blocks_new_claim(self):
        for name in ('_failed_sealed_test_example','.v5-sealed-test-leftover','sealed_test'):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);(out/name).mkdir()
                with self.assertRaises(PermissionError):claim_attempt(out,'abc')

    def test_full_freeze_checks_source_scaler_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'code').write_bytes(b'code');(root/'scaler').write_bytes(b'scaler')
            f={'status':'FROZEN_FOR_ONE_SEALED_TEST','test_unlock':True,'allowed_runs':1,
               'source_sha256':{'code':file_sha256(root/'code')},
               'artifact_sha256':{'scaler':file_sha256(root/'scaler')},'runtime_versions':runtime_versions()}
            path=root/'freeze.json';path.write_text(json.dumps(f));verify_freeze(path,root,False)
            (root/'code').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_freeze(path,root,False)
            (root/'code').write_bytes(b'code');(root/'scaler').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_freeze(path,root,False)
            (root/'scaler').write_bytes(b'scaler');f['runtime_versions']['torch']='wrong'
            path.write_text(json.dumps(f))
            with self.assertRaises(ValueError):verify_freeze(path,root,False)

    def test_simultaneous_claim_has_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            def contender(_):
                try: claim_attempt(Path(tmp),'abc');return 1
                except (PermissionError,FileExistsError):return 0
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(pool.map(contender,range(2))),1)

    def test_intent_exact_cutoff_uses_no_future_to_select(self):
        evaluator=script('intent_fixture','scripts/57_evaluate_v5_models.py')
        base=script('metric_fixture','scripts/15_train_rssm.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];labels=[]
            for family in range(4):
                for malicious in (False,True):
                    eid=f'fixture_{family}_{malicious}';p=root/eid;p.mkdir()
                    pd.DataFrame([{'forecast_time':'2026-01-01T00:01:40+00:00'}]).to_csv(p/'episode_metadata.csv',index=False)
                    for t in (95,100,105):
                        rows.append({'episode_id':eid,'cohort':'intent_probe','paired_family':str(family),
                            'scenario':'credential_one_hop' if malicious else 'matched_legitimate_ssh',
                            'prediction_available_time':str(pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(seconds=t)),
                            'lateral_movement_already_observed':0,'context_first_state':t//5-3})
                        labels.append(int(malicious))
            frame=pd.DataFrame(rows);data={'lateral_movement_within_horizon':np.array(labels)}
            probs=np.full(len(frame),.1)
            result=evaluator.intent_probe(base,frame,data,probs,.5,root)
            self.assertEqual(result['contexts'],8)
            self.assertEqual(result['mean_malicious_minus_legitimate_probability'],0)
            self.assertTrue(all('00:01:40' in r['forecast_time'] for r in result['episode_rows']))
            frame.loc[1,'prediction_available_time']='2026-01-01T00:01:45+00:00'
            with self.assertRaises(ValueError):evaluator.intent_probe(base,frame,data,probs,.5,root)

    def test_direct_test_evaluator_fails_before_bundle_loading(self):
        evaluator=script('direct_guard','scripts/57_evaluate_v5_models.py')
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(evaluator,'verify_freeze',side_effect=PermissionError('no freeze')),
              patch.object(evaluator,'load_bundle') as read_bundle,
              patch.object(evaluator.torch.cuda,'is_available',return_value=True)):
            with patch.object(sys,'argv',['evaluate','--split','test','--out',str(Path(tmp)/'report.json')]):
                with self.assertRaises(PermissionError):evaluator.main()
            read_bundle.assert_not_called()


if __name__=='__main__':unittest.main()
