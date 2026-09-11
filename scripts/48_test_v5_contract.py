#!/usr/bin/env python3
"""Synthetic V5 export/leakage regressions plus untrained CUDA graph checks."""
from __future__ import annotations
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import HOSTS, IPS, PAIRS, GLOBAL, NODE, EDGE, roles, schedule, feature_metadata
from src.cyberwm.v5_sequences import episode_samples, load_observable_states, permutation_indices
from src.cyberwm.action_graph_rssm import ActionGraphRSSM
from src.cyberwm.graph_rssm import fit_graph_feature_scaler
spec = importlib.util.spec_from_file_location("export_v5", ROOT/"scripts/47_build_v5_sequences.py")
exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)


def fixture(ep, action="block_ssh"):
    ep.mkdir(); (ep/"states").mkdir()
    starts = pd.date_range("2000-01-01T00:00:05Z", periods=29, freq="5s")
    actor, pivot, target, b1, b2 = roles(90002)
    gs=[]; ns=[]; es=[]
    for i, t in enumerate(starts):
        g={"state_id": i, "window_start": str(t), **dict.fromkeys(GLOBAL, 0)}
        if i % 3 == 0:
            g.update(flow_count=1, unique_hosts=2, unique_edges=1, total_bytes=64, total_packets=1)
            for host in [HOSTS[actor], HOSTS[pivot]]:
                ns.append({"state_id": i, "window_start": str(t), "host": host, **dict.fromkeys(NODE[:-1], 0), "is_internal": 1})
            es.append({"state_id": i, "window_start": str(t), "source_ip": HOSTS[actor], "destination_ip": HOSTS[pivot],
                       **dict.fromkeys(EDGE[:-1], 0), "flow_count": 1, "bytes_total": 64, "packets_total": 1, "internal_edge": True})
        gs.append(g)
    pd.DataFrame(gs).to_csv(ep/"states/global_states.csv", index=False)
    pd.DataFrame(ns).to_csv(ep/"states/node_states.csv.gz", index=False)
    pd.DataFrame(es).to_csv(ep/"states/edge_states.csv.gz", index=False)
    th = "episode_id start_time end_time actor target technique_id technique tactic".split()
    truth=[]
    if action != "none":
        truth=[dict(zip(th, [ep.name, "2000-01-01T00:01:36.4Z", "2000-01-01T00:01:36.4Z", actor, pivot,
                             "T1021.004", "SSH", "Lateral Movement" if action == "permit_ssh" else "Lateral Movement Attempt"]))]
    pd.DataFrame(truth, columns=th).to_csv(ep/"ground_truth.csv", index=False)
    aligned = pd.DataFrame({"state_id": np.arange(29), "window_start": starts.astype(str),
                           "technique_ids": "", "tactics": "", "actors": "", "targets": "", "has_lateral_movement": 0})
    if action != "none":
        aligned.loc[18, "technique_ids"] = "T1021.004"
        aligned.loc[18, "has_lateral_movement"] = int(action == "permit_ssh")
    aligned.to_csv(ep/"state_ground_truth.csv", index=False)
    meta=dict(episode_id=ep.name, scenario="credential_action_block" if action != "none" else "background_only",
              seed=90002, capture_start="2000-01-01T00:00:01Z", capture_end="2000-01-01T00:02:31Z",
              actor=actor,pivot=pivot,target=target,background_host_1=b1,background_host_2=b2,
              background_profile="mixed",topology_profile="flat_five_host",node_count=5,window_seconds=5,
              planned_capture_duration_seconds=150,defender_action=action,runtime_version="v5_reviewed_1",
              forecast_time="2000-01-01T00:01:35Z")
    pd.DataFrame([meta]).to_csv(ep/"episode_metadata.csv", index=False)
    ah="episode_id start_time end_time action source target known_at_forecast_time details forecast_time effective_start_time".split()
    ar=[] if action == "none" else [dict(zip(ah,[ep.name,"2000-01-01T00:01:34Z","2000-01-01T00:01:35.4Z",action,actor,pivot,
                                              "true","test",meta["forecast_time"],"2000-01-01T00:01:35.2Z"]))]
    pd.DataFrame(ar,columns=ah).to_csv(ep/"defender_actions.csv",index=False)
    return meta


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.ep=self.root/"v5_smoke_fixture"
        self.meta=fixture(self.ep)
    def tearDown(self): self.tmp.cleanup()
    def test_shapes_attempt_and_action(self):
        a,m=episode_samples(self.ep,"action")
        self.assertEqual(a["context_states"].shape,(1,3,345)); self.assertEqual(a["future_states"].shape,(1,6,345))
        self.assertEqual(a["future_edge_presence"].shape,(1,6,20)); self.assertEqual(a["action_pair"].sum(),1)
        self.assertEqual(a["future_techniques"][0,0,2],1); self.assertEqual(a["lateral_movement_within_horizon"].sum(),0)
        passive,_=episode_samples(self.ep,"passive_action")
        self.assertNotIn("action_type",passive); np.testing.assert_array_equal(passive["context_states"],a["context_states"])
        self.assertEqual(m[0]["context_last_state"],17)
    def test_positive_point_event(self):
        ep=self.root/"positive"; fixture(ep,"permit_ssh")
        a,_=episode_samples(ep,"action"); self.assertEqual(a["future_lateral_edges"].sum(),1)
        self.assertEqual(a["lateral_movement_within_horizon"].sum(),1)
    def test_passive_windows_and_silent_inventory(self):
        ep=self.root/"benign";fixture(ep,"none")
        a,_=episode_samples(ep,"passive"); self.assertEqual(a["context_states"].shape,(21,3,345))
        states,_,_=load_observable_states(ep)
        self.assertTrue((states[1,15:105].reshape(5,18)[:,-1]==0).all())
        self.assertTrue((states[1,15:105].reshape(5,18)[:,NODE.index("is_internal")]==1).all())
    def test_truth_and_future_independence(self):
        before=load_observable_states(self.ep)[0].copy()
        (self.ep/"ground_truth.csv").write_text("deliberately unusable truth")
        np.testing.assert_array_equal(before,load_observable_states(self.ep)[0])
        g=pd.read_csv(self.ep/"states/global_states.csv");g.loc[20:,"total_bytes"]+=999
        g.to_csv(self.ep/"states/global_states.csv",index=False)
        np.testing.assert_array_equal(before[:18],load_observable_states(self.ep)[0][:18])
    def test_late_action_and_partial_future_rejected(self):
        p=self.ep/"defender_actions.csv";d=pd.read_csv(p, dtype=str)
        d.loc[0,"start_time"]="2000-01-01T00:01:35.1Z";d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,"not known"):episode_samples(self.ep,"action")
        d.loc[0,"start_time"]="2000-01-01T00:01:34Z";d.loc[0,"forecast_time"]="2000-01-01T00:02:20Z"
        d.loc[0,"effective_start_time"]="2000-01-01T00:02:20.2Z";d.loc[0,"end_time"]="2000-01-01T00:02:20.4Z";d.to_csv(p,index=False)
        meta=pd.read_csv(self.ep/"episode_metadata.csv");meta.loc[0,"forecast_time"]=d.loc[0,"forecast_time"];meta.to_csv(self.ep/"episode_metadata.csv",index=False)
        with self.assertRaisesRegex(ValueError,"insufficient complete"):episode_samples(self.ep,"action")
    def test_unknown_hosts_and_extra_labels_rejected(self):
        p=self.ep/"states/node_states.csv.gz"; d=pd.read_csv(p);d.loc[0,"host"]="10.77.0.99";d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,"unknown observable"):load_observable_states(self.ep)
        g=pd.read_csv(self.ep/"states/global_states.csv");g["Label"]=0;g.to_csv(self.ep/"states/global_states.csv",index=False)
        with self.assertRaisesRegex(ValueError,"unexpected columns"):load_observable_states(self.ep)
    def test_temporal_reference_and_finiteness(self):
        p=self.ep/"states/edge_states.csv.gz";d=pd.read_csv(p);d.loc[0,"state_id"]=99;d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,"reference"):load_observable_states(self.ep)
    def test_test_lock_before_io_and_atomic_export(self):
        plan=pd.DataFrame([{**self.meta,"split":"smoke","cohort":"smoke","paired_family":"fixture"},
                           {**self.meta,"episode_id":"must_not_read","split":"test","cohort":"test","paired_family":"sealed"}])
        with self.assertRaises(PermissionError):exporter.export(plan,"test","action",self.root,self.root/"locked")
        self.assertFalse((self.root/"locked").exists())
        exporter.export(plan,"smoke","action",self.root,self.root/"out")
        with self.assertRaises(FileExistsError):exporter.export(plan,"smoke","action",self.root,self.root/"out")
        with np.load(self.root/"out/smoke.npz") as a:self.assertEqual(a["action_pair"].shape,(1,20))
    def test_all_host_permutations(self):
        state=np.arange(345); pair=np.arange(20)
        for order in permutations(range(5)):
            si,ei=permutation_indices(order); invs,inve=permutation_indices(np.argsort(order))
            np.testing.assert_array_equal(state[si][invs],state);np.testing.assert_array_equal(pair[ei][inve],pair)
    def test_capture_freeze_gate(self):
        spec=importlib.util.spec_from_file_location("freeze_v5_test",ROOT/"scripts/49_freeze_v5_capture.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.FREEZE=self.root/"absent_freeze.json"
        with self.assertRaisesRegex(PermissionError,"NOT FROZEN"):module.check_freeze()
    def test_schedule_bounds(self):
        schedules=[schedule(i) for i in range(10000,10200)]
        self.assertGreater(len({s["decision_seconds"] for s in schedules}),10)
        for s in schedules:self.assertLessEqual(s["decision_seconds"]+5+1+8+30,150)


def model_checks():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(48001); torch.set_num_threads(4)
    model=ActionGraphRSSM(node_count=5,pair_count=20).to(device)
    original=ActionGraphRSSM(); model.load_state_dict(original.state_dict(),strict=True)
    x=torch.randn(4,3,345,device=device); action=torch.eye(2,device=device).repeat(2,1)
    pair=torch.eye(20,device=device)[[0,3,7,19]]
    with torch.no_grad():
        base=model.forecast_action(x,action,pair)
        delta=0.0
        for order in permutations(range(5)):
            si,ei=permutation_indices(order)
            other=model.forecast_action(x[...,si],action,pair[...,ei])
            delta=max(delta,float((other["decoded"]-base["decoded"][...,si]).abs().max()),
                      float((other["pair_logits"]-base["pair_logits"][...,ei]).abs().max()),
                      float((other["lm_logits"]-base["lm_logits"]).abs().max()))
        assert delta<2e-6,delta
        full=torch.cat([x,torch.randn(4,6,345,device=device)],1)
        _,a=model.forward_action(full,3,action,pair,sample=False)
        full[:,3:]+=1000
        _,b=model.forward_action(full,3,action,pair,sample=False)
        cuda_causal_delta=float((a["decoded"]-b["decoded"]).abs().max())
        assert cuda_causal_delta < 1e-6, cuda_causal_delta
        # CUDA index_add reductions can differ at ~1e-8 even on identical
        # inputs. Verify exact temporal independence separately on CPU.
        cpu_model=ActionGraphRSSM(node_count=5,pair_count=20)
        cpu_model.load_state_dict(model.state_dict())
        cpu_full=full.cpu().clone()
        _,ca=cpu_model.forward_action(cpu_full,3,action.cpu(),pair.cpu(),sample=False)
        cpu_full[:,3:]-=1000
        _,cb=cpu_model.forward_action(cpu_full,3,action.cpu(),pair.cpu(),sample=False)
        assert torch.equal(ca["decoded"],cb["decoded"])
    prediction=model.forecast_action(x,action,pair,sample=True)
    (prediction["decoded"].square().mean()+prediction["lm_logits"].square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    raw=np.random.default_rng(48).normal(size=(12,3,345)).astype(np.float32)
    scaler=fit_graph_feature_scaler(raw,15,18,5,12,20)
    si,_=permutation_indices([4,2,1,0,3])
    np.testing.assert_allclose(scaler.transform(raw.reshape(-1,345))[...,si],scaler.transform(raw[...,si].reshape(-1,345)),atol=1e-6)
    # Inspect/load only parameter dictionaries, never prior test examples or
    # predictions. Node/pair buffers are regenerated for five hosts.
    source_checks={}
    for filename in ["friday_graph_rssm_pretrained_fixed.pt", "mvp_v4_action_graph_rssm_scratch.pt"]:
        path=ROOT/"models"/filename
        if not path.is_file():
            source_checks[filename]="absent: not tested"; continue
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        config={**checkpoint["model_config"],"node_count":5,"pair_count":20}
        transferred=ActionGraphRSSM(**config)
        loaded=transferred.load_state_dict(checkpoint["model_state_dict"],strict=False)
        assert not loaded.unexpected_keys
        assert all(k.startswith("action_") for k in loaded.missing_keys)
        assert len(checkpoint["scaler_mean"])==141  # expressly NOT reused for 345 features
        source_checks[filename]={"missing_action_parameters":len(loaded.missing_keys),"old_scaler_reused":False}
    print(json.dumps({"device":str(device),"frozen_parameter_shape_checks":source_checks,"all_120_permutations_max_delta":delta,
                      "cpu_future_perturbation_delta":0,"cuda_future_perturbation_delta":cuda_causal_delta,"finite_gradients":True,
                      "three_to_five_parameter_shapes_compatible":True}))


if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():sys.exit(1)
    model_checks()
