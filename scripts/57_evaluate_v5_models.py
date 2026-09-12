#!/usr/bin/env python3
"""Evaluate frozen V5 models on validation smoke or one sealed test bundle."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM
from src.cyberwm.branch_metrics import branch_forecast_metrics, calibration_metrics
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM
from src.cyberwm.v5_model_protocol import file_sha256, load_scaler
from src.cyberwm.v5_contract import feature_metadata
from src.cyberwm.v5_evaluation_seal import (FREEZE, CLAIM, ROOT as SEAL_ROOT, checkpoint_contract,
    verify_freeze, claim_inference, runtime_versions)

DEVICE = torch.device('cuda')
REGIMES = ("scratch", "friday", "v4")


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def load_bundle(root: Path, mode: str, split: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    directory = root / mode / split
    with np.load(directory / f"{split}.npz") as loaded: arrays = {name: loaded[name] for name in loaded.files}
    manifest = pd.read_csv(directory / "sample_manifest.csv")
    metadata = json.loads((directory / "feature_metadata.json").read_text())
    if set(manifest.split) != {split} or metadata["split"] != split or metadata["mode"] != mode:
        raise ValueError(f"bundle marker mismatch {mode}/{split}")
    for key, expected in feature_metadata().items():
        if metadata.get(key) != expected: raise ValueError(f'feature contract differs: {key}')
    plan = pd.read_csv(ROOT / 'configs/mvp_v5_episode_plan.csv', dtype=str)
    selected = plan[plan.split.eq(split)]
    selected = selected[selected.defender_action.eq('none') if mode == 'passive' else selected.defender_action.ne('none')]
    if set(manifest.episode_id) != set(selected.episode_id): raise ValueError('bundle episode membership')
    if set(metadata['source_hashes']) != set(selected.episode_id): raise ValueError('source membership')
    if metadata.get('test_access') is not (split=='test'): raise ValueError('test access marker')
    if not np.array_equal(manifest.sample_id,np.arange(len(manifest))): raise ValueError('sample IDs/order')
    for episode_id, rows in manifest.groupby('episode_id'):
        if len(rows)!=(21 if mode=='passive' else 1): raise ValueError('windows per episode')
        if mode=='passive' and rows.future_first_state.tolist()!=list(range(3,24)):
            raise ValueError('passive windows missing/duplicated')
        for relative,digest in metadata['source_hashes'][episode_id].items():
            if file_sha256(ROOT/'lab/episodes'/episode_id/relative)!=digest: raise ValueError('source changed since export')
    base_shapes = dict(context_states=(3,345), future_states=(6,345), future_edge_presence=(6,20),
        future_lateral_movement=(6,), future_techniques=(6,3), future_lateral_edges=(6,20),
        lateral_movement_within_horizon=())
    if mode == 'action': base_shapes.update(action_type=(2,), action_pair=(20,))
    if set(arrays) != set(base_shapes): raise ValueError('unexpected model arrays')
    for name, shape in base_shapes.items():
        value = arrays[name]
        if value.shape != (len(manifest), *shape) or not np.isfinite(value).all(): raise ValueError(f'invalid array {name}')
        if name not in ('context_states','future_states') and not np.isin(value,[0,1]).all(): raise ValueError('nonbinary target/action')
    if not np.array_equal(arrays['future_lateral_edges'].max(2), arrays['future_lateral_movement']): raise ValueError('LM pair alignment')
    if not np.array_equal(arrays['future_lateral_movement'].max(1), arrays['lateral_movement_within_horizon']): raise ValueError('LM horizon alignment')
    for row in manifest.itertuples():
        if row.context_last_state-row.context_first_state != 2 or row.future_first_state != row.context_last_state+1 or row.future_last_state-row.future_first_state != 5:
            raise ValueError('context/future indexing')
        planned = selected[selected.episode_id.eq(row.episode_id)].iloc[0]
        for key in ['scenario','cohort','paired_family','background_profile']:
            if getattr(row,key) != planned[key]: raise ValueError(f'manifest plan mismatch {key}')
    if mode == 'action':
        if not (arrays['action_type'].sum(1)==1).all() or not (arrays['action_pair'].sum(1)==1).all(): raise ValueError('action one-hot')
    return arrays, manifest, metadata


def forecast_cpu(model, method, *inputs):
    chunks = []
    with torch.no_grad():
        for start in range(0,len(inputs[0]),128):
            output = getattr(model,method)(*[x[start:start+128].to(DEVICE) for x in inputs], sample=False)
            chunks.append({k:v.cpu() for k,v in output.items() if k in
                ('decoded','edge_logits','lm_logits','technique_logits','pair_logits','branch_weights')})
    return {k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}


def state_metrics(predicted: np.ndarray, target: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    absolute = np.abs(predicted-target); active = raw != 0
    return {"overall_mae": float(absolute.mean()), "global_mae": float(absolute[..., :15].mean()),
            "node_mae": float(absolute[..., 15:105].mean()), "edge_mae": float(absolute[..., 105:].mean()),
            "active_mae": float(absolute[active].mean()) if active.any() else None,
            "quiet_mae": float(absolute[~active].mean()) if (~active).any() else None,
            "mae_by_5s_horizon": absolute.mean(axis=(0,2)).tolist()}


def binary_report(module: Any, labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    return {"fixed_0_5": module.binary_metrics(labels, probabilities, 0.5),
            "frozen_validation_threshold": threshold,
            "at_frozen_validation_threshold": module.binary_metrics(labels, probabilities, threshold),
            "ranking_and_calibration_diagnostic": calibration_metrics(labels, probabilities)}


def technique_report(target: np.ndarray, probability: np.ndarray, names: list[str]) -> dict[str, Any]:
    target = target.max(axis=1).astype(int)
    rows = {}
    for index, name in enumerate(names):
        labels = target[:, index]
        rows[name] = {"positive_samples": int(labels.sum()),
                      "average_precision": float(average_precision_score(labels, probability[:, index]))
                      if 0 < labels.sum() < len(labels) else None,
                      "mean_probability": float(probability[:, index].mean())}
    return {"micro_average_precision": float(average_precision_score(target.ravel(), probability.ravel())),
            "techniques": rows}


def scenario_metrics(module: Any, manifest: pd.DataFrame, target: np.ndarray, predicted: np.ndarray,
                     labels: np.ndarray, probabilities: np.ndarray, threshold: float, column='scenario') -> dict[str, Any]:
    rows = {}
    for scenario in sorted(manifest[column].unique()):
        mask = manifest[column].eq(scenario).to_numpy(); y = labels[mask]; p = probabilities[mask]
        rows[scenario] = {"samples": int(mask.sum()), "state_mae": float(np.abs(predicted[mask]-target[mask]).mean()),
                          "lm_positive": int(y.sum()), "lm_mean_probability": float(p.mean()),
                          "lm_f1_fixed_0_5": module.binary_metrics(y, p, 0.5)["f1"],
                          "lm_f1_frozen_validation_threshold": module.binary_metrics(y, p, threshold)["f1"],
                          "lm_average_precision": float(average_precision_score(y,p)) if 0 < y.sum() < len(y) else None}
    return rows


def persistence(module: Any, data: dict[str, np.ndarray], scaler: Any) -> dict[str, Any]:
    context = module.normalize(scaler, data["context_states"]); target = module.normalize(scaler, data["future_states"])
    predicted = np.repeat(context[:, -1:, :], 6, axis=1)
    edge_last = data["context_states"][:, -1, 105:].reshape(len(context), 20, 12)[:, :, -1]
    edges = np.repeat(edge_last[:, None, :], 6, axis=1)
    return {"state": state_metrics(predicted, target, data["future_states"]),
            "future_edge_average_precision": float(average_precision_score(data["future_edge_presence"].ravel(), edges.ravel())),
            "lm_policy": "always zero; F1 is zero when positives exist"}


def evaluate_action(module: Any, data: dict[str, np.ndarray], manifest: pd.DataFrame, scaler: Any,
                    thresholds: dict[str, float], checkpoint_hashes: dict[str, str], saved=None) -> dict[str, Any]:
    context = torch.from_numpy(module.normalize(scaler, data["context_states"])); target = module.normalize(scaler, data["future_states"])
    action_type = torch.from_numpy(data["action_type"]); action_pair = torch.from_numpy(data["action_pair"])
    labels = data["lateral_movement_within_horizon"].astype(int); results = {}
    for regime in REGIMES:
        path = ROOT / "models" / f"mvp_v5_action_graph_rssm_{regime}.pt"
        if file_sha256(path) != checkpoint_hashes[regime]: raise ValueError(f"action hash {regime}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActionGraphRSSM(**checkpoint["model_config"]).to(DEVICE); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        factual = forecast_cpu(model, 'forecast_action', context, action_type, action_pair)
        opposite = forecast_cpu(model, 'forecast_action', context, action_type.flip(1), action_pair)
        permit = forecast_cpu(model, 'forecast_action', context, torch.tensor([[1.,0.]]).expand(len(context),-1), action_pair)
        block = forecast_cpu(model, 'forecast_action', context, torch.tensor([[0.,1.]]).expand(len(context),-1), action_pair)
        if saved is not None:
            saved.update({f'action_{regime}_{k}':v.numpy() for k,v in factual.items()})
            saved[f'action_{regime}_opposite_decoded'] = opposite['decoded'].numpy()
        decoded = factual["decoded"].numpy(); edge = torch.sigmoid(factual["edge_logits"]).numpy()
        lm = torch.sigmoid(factual["lm_logits"]).numpy(); pair = torch.sigmoid(factual["pair_logits"]).numpy()
        technique = torch.sigmoid(factual["technique_logits"]).numpy()
        pair_target = data["future_lateral_edges"].max(axis=1); positive = pair_target.sum(1)>0
        factual_error = np.abs(decoded-target).mean((1,2)); opposite_error = np.abs(opposite["decoded"].numpy()-target).mean((1,2))
        threshold = thresholds[regime]
        results[regime] = {"state": state_metrics(decoded,target,data["future_states"]),
            "future_edge_average_precision": float(average_precision_score(data["future_edge_presence"].ravel(),edge.ravel())),
            "lm": binary_report(module,labels,lm,threshold),
            "attack_techniques": technique_report(data["future_techniques"], technique, ["T1046","T1110.001","T1021.004"]),
            "pair": {"micro_average_precision": float(average_precision_score(pair_target.ravel(),pair.ravel())),
                     "top1_positive_samples": float(np.mean(pair_target[positive,pair[positive].argmax(1)]>0)),
                     "positive_samples": int(positive.sum())},
            "counterfactual": {"factual_lower_state_error_count": int((factual_error<opposite_error).sum()),
                "contexts": len(factual_error), "mean_factual_minus_opposite_state_error": float((factual_error-opposite_error).mean()),
                "mean_permit_minus_block_lm_probability": float((torch.sigmoid(permit["lm_logits"])-torch.sigmoid(block["lm_logits"])).mean())},
            "by_scenario": scenario_metrics(module,manifest,target,decoded,labels,lm,threshold),
            "by_background_profile": scenario_metrics(module,manifest,target,decoded,labels,lm,threshold,'background_profile'),
            "by_episode": scenario_metrics(module,manifest,target,decoded,labels,lm,threshold,'episode_id'),
            "edge_positive_rate": float(data['future_edge_presence'].mean())}
    return {"models": results, "persistence": persistence(module,data,scaler)}


def evaluate_passive(module: Any, data: dict[str, np.ndarray], manifest: pd.DataFrame, metadata: dict[str, Any],
                     scaler: Any, thresholds: dict[str,float], checkpoint_hashes: dict[str,str], saved=None, prefix='passive'
                     ) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    context = torch.from_numpy(module.normalize(scaler,data["context_states"])); target=module.normalize(scaler,data["future_states"])
    labels=data["lateral_movement_within_horizon"].astype(int); results={}; probabilities={}
    groups={"global":slice(0,15),"node":slice(15,105),"edge":slice(105,345)}
    for regime in REGIMES:
        path=ROOT/"models"/f"mvp_v5_passive_branch_{regime}.pt"
        if file_sha256(path)!=checkpoint_hashes[regime]: raise ValueError(f"passive hash {regime}")
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        model=BranchingGraphRSSM(**checkpoint["model_config"]).to(DEVICE);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
        output=forecast_cpu(model,'forecast_branches',context)
        if saved is not None: saved.update({f'{prefix}_{regime}_{k}':v.numpy() for k,v in output.items()})
        decoded=output["decoded"].numpy();weights=output["branch_weights"].numpy();lm=weights[:,1];probabilities[regime]=lm
        edge=torch.sigmoid(output["edge_logits"]).numpy()
        future=branch_forecast_metrics(decoded.transpose(1,0,2,3),target,weights,groups,edge.transpose(1,0,2,3),data["future_edge_presence"])
        expected=np.einsum("nb,nbhf->nhf",weights,decoded)
        pair_branches=torch.sigmoid(output["pair_logits"]).numpy();pair=(weights[...,None]*pair_branches).sum(1)
        technique_branches=torch.sigmoid(output["technique_logits"]).numpy();technique=(weights[...,None]*technique_branches).sum(1)
        pair_target=data["future_lateral_edges"].max(1);positive=pair_target.sum(1)>0;threshold=thresholds[regime]
        results[regime]={"future_graph":future,"expected_state":state_metrics(expected,target,data["future_states"]),
            "lm":binary_report(module,labels,lm,threshold),
            "attack_techniques":technique_report(data["future_techniques"],technique,["T1046","T1110.001","T1021.004"]),
            "pair":{"micro_average_precision":float(average_precision_score(pair_target.ravel(),pair.ravel())),
                    "top1_positive_samples":float(np.mean(pair_target[positive,pair[positive].argmax(1)]>0)) if positive.any() else None,
                    "positive_samples":int(positive.sum())},
            "by_scenario":scenario_metrics(module,manifest,target,expected,labels,lm,threshold),
            "by_background_profile":scenario_metrics(module,manifest,target,expected,labels,lm,threshold,'background_profile'),
            "by_episode":scenario_metrics(module,manifest,target,expected,labels,lm,threshold,'episode_id')}
    return {"models":results,"persistence":persistence(module,data,scaler)},probabilities


def intent_probe(module: Any, manifest: pd.DataFrame, data: dict[str,np.ndarray], probability: np.ndarray,
                 threshold: float, episodes_dir: Path) -> dict[str,Any]:
    mask=manifest.cohort.eq("intent_probe"); rows=[]
    for episode_id in sorted(manifest.loc[mask,"episode_id"].unique()):
        candidates=manifest.index[manifest.episode_id.eq(episode_id)].to_numpy()
        meta=pd.read_csv(episodes_dir/episode_id/'episode_metadata.csv').iloc[0]
        forecast=pd.Timestamp(meta.forecast_time)
        if forecast.tzinfo is None or forecast != forecast.floor('5s'): raise ValueError('intent cutoff UTC/grid')
        times=pd.to_datetime(manifest.loc[candidates,"prediction_available_time"],utc=True)
        exact=np.flatnonzero(times==forecast)
        if len(exact)!=1: raise ValueError(f"intent forecast alignment {episode_id}: {len(exact)}")
        index=int(candidates[exact[0]]); row=manifest.loc[index]
        if row.lateral_movement_already_observed != 0: raise ValueError('intent probe already observed LM')
        rows.append({"episode_id":episode_id,"paired_family":row.paired_family,"scenario":row.scenario,
                     "forecast_time":str(forecast), "context_first_state":int(row.context_first_state),
                     "label":int(data["lateral_movement_within_horizon"][index]),"lm_probability":float(probability[index])})
    frame=pd.DataFrame(rows)
    if len(frame)!=8 or frame.paired_family.nunique()!=4: raise ValueError('intent probe coverage')
    labels=frame.label.to_numpy();probs=frame.lm_probability.to_numpy()
    differences=[]
    for family,group in frame.groupby("paired_family"):
        if len(group)!=2: raise ValueError(f"intent family size {family}")
        malicious=group[group.scenario.eq("credential_one_hop")];legitimate=group[group.scenario.eq("matched_legitimate_ssh")]
        if len(malicious)!=1 or len(legitimate)!=1: raise ValueError(f"intent scenarios {family}")
        if malicious.label.iloc[0]!=1 or legitimate.label.iloc[0]!=0: raise ValueError('intent horizon truth mismatch')
        differences.append(float(malicious.lm_probability.iloc[0]-legitimate.lm_probability.iloc[0]))
    return {"contexts":len(frame),"episode_rows":frame.to_dict("records"),
            "classification":binary_report(module,labels,probs,threshold),
            "malicious_minus_legitimate_probability_by_family":differences,
            "mean_malicious_minus_legitimate_probability":float(np.mean(differences))}


def episode_warnings(manifest, probability, threshold, episodes_dir):
    rows=[]
    for episode_id, group in manifest.groupby('episode_id', sort=True):
        truth=pd.read_csv(episodes_dir/episode_id/'ground_truth.csv')
        events=truth[truth.tactic.eq('Lateral Movement')]
        onset=pd.to_datetime(events.start_time,utc=True,format='mixed').min() if len(events) else None
        times=pd.to_datetime(group.prediction_available_time,utc=True)
        p=probability[group.index.to_numpy()]
        alert=p>=threshold
        before=times<onset if onset is not None else np.ones(len(group),dtype=bool)
        in_horizon=before & ((times+pd.Timedelta(seconds=30)>onset) if onset is not None else False)
        useful=alert & in_horizon
        first=times[useful].min() if np.any(useful) else None
        rows.append({'episode_id':episode_id,'scenario':group.scenario.iloc[0],
            'has_completed_lm':onset is not None,'pre_first_or_negative_alert':bool(np.any(alert & before)),
            'within_30s_pre_first_detection':bool(np.any(useful)),
            'lead_seconds_to_successful_action_start':float((onset-first).total_seconds()) if first is not None else None,
            'premature_alert_before_30s_horizon':bool(np.any(alert & before & ~in_horizon)) if onset is not None else None})
    return {'threshold':threshold,'episodes':rows,
        'warning':'lead is to controller-recorded successful SSH action start, not an independently timed compromise completion; overlapping windows are not independent trials'}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--split",choices=["validation","test"],required=True)
    ap.add_argument("--sequences-dir",type=Path,default=ROOT/"outputs/mvp_v5/sequences")
    ap.add_argument("--evaluation-freeze",type=Path,default=ROOT/"configs/mvp_v5_evaluation_freeze.json")
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--episodes-dir",type=Path,default=ROOT/"lab/episodes")
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if not torch.cuda.is_available(): raise RuntimeError('reviewed evaluator requires CUDA')
    torch.set_num_threads(8)
    freeze=None
    if args.split=='test':
        if args.evaluation_freeze.resolve()!=FREEZE.resolve() or args.episodes_dir.resolve()!=(ROOT/'lab/episodes').resolve():
            raise PermissionError('noncanonical sealed evaluation paths')
        freeze=verify_freeze()
        claim_inference(CLAIM,file_sha256(FREEZE),args.sequences_dir,args.out)
        checkpoint_hashes=freeze['checkpoint_sha256']; thresholds=freeze['thresholds']
        primary_action=freeze['primary_action']; primary_passive=freeze['primary_passive']
    else:
        if args.sequences_dir.resolve()!=(ROOT/'outputs/mvp_v5/sequences').resolve():
            raise PermissionError('validation smoke must use canonical validation bundles')
        checkpoint_hashes,thresholds,_,_=checkpoint_contract()
        validation=json.loads((ROOT/'outputs/mvp_v5/validation_audit/audit.json').read_text())
        primary_action=validation['action_primary']; primary_passive=validation['passive_primary']
    action,action_manifest,action_meta=load_bundle(args.sequences_dir,"action",args.split)
    aligned,aligned_manifest,aligned_meta=load_bundle(args.sequences_dir,"passive_action",args.split)
    if not action_manifest.equals(aligned_manifest): raise ValueError('action/aligned manifests differ')
    for name in aligned:
        if not np.array_equal(action[name],aligned[name]):raise ValueError(f"action/aligned mismatch {name}")
    passive,passive_manifest,passive_meta=load_bundle(args.sequences_dir,"passive",args.split)
    expected={"validation":(8,168,8),"test":(8,336,8)}[args.split]
    if (len(action["context_states"]),len(passive["context_states"]),len(aligned["context_states"]))!=expected:raise ValueError("sample count")
    scaler_path=ROOT/"outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz";scaler=load_scaler(scaler_path)
    module=load_script("v5_sealed_base",ROOT/"scripts/15_train_rssm.py")
    saved={}
    action_result=evaluate_action(module,action,action_manifest,scaler,thresholds['action'],checkpoint_hashes['action'],saved)
    passive_result,passive_probability=evaluate_passive(module,passive,passive_manifest,passive_meta,scaler,thresholds['passive'],checkpoint_hashes['passive'],saved)
    aligned_result,_=evaluate_passive(module,aligned,aligned_manifest,aligned_meta,scaler,thresholds['passive'],checkpoint_hashes['passive'],saved,'aligned')
    report={"status":"SEALED_TEST_COMPLETE" if args.split=="test" else "VALIDATION_EVALUATOR_SMOKE_COMPLETE",
            "split":args.split,"test_access":args.split=="test","primary_action":primary_action,
            "primary_passive":primary_passive,"scaler_sha256":file_sha256(scaler_path),
            "evaluator_sha256":file_sha256(Path(__file__).resolve()),'device':str(DEVICE),
            'episode_warning_primary':{name:episode_warnings(passive_manifest,passive_probability[primary_passive],threshold,args.episodes_dir)
                for name,threshold in [('fixed_0_5',0.5),('frozen_validation_threshold',thresholds['passive'][primary_passive])]},
            "checkpoint_sha256":checkpoint_hashes,"thresholds":thresholds,
            "runtime_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,
                                "sklearn":sklearn.__version__,"torch":torch.__version__,"torch_cuda":torch.version.cuda},
            "action":action_result,
            "passive":passive_result,"passive_on_action_aligned":aligned_result,
            "oracle_warning":"oracle branch metrics measure candidate coverage, not deployable inference accuracy",
            "calibration_warning":"Brier/ECE on small correlated samples are diagnostics only"}
    if args.split=="test":
        primary=primary_passive
        report["test_only_intent_probe"]=intent_probe(module,passive_manifest,passive,passive_probability[primary],
                                                       thresholds["passive"][primary],args.episodes_dir)
        report["evaluation_freeze_sha256"]=file_sha256(args.evaluation_freeze)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    forecast_path=args.out.with_suffix('.predictions.npz')
    if forecast_path.exists(): raise FileExistsError(forecast_path)
    np.savez_compressed(forecast_path,**saved)
    report['predictions_sha256']=file_sha256(forecast_path)
    temp=Path(tempfile.mkdtemp(prefix='.v5-eval-',dir=args.out.parent))
    try:
        path=temp/args.out.name;path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");os.rename(path,args.out)
    finally:shutil.rmtree(temp,ignore_errors=True)
    print(json.dumps({"status":report["status"],"split":args.split,
        "primary_action_state_mae":action_result["models"][report["primary_action"]]["state"]["overall_mae"],
        "primary_passive_expected_mae":passive_result["models"][report["primary_passive"]]["future_graph"]["expected_forecast_mae"],
        "primary_action_lm_f1":action_result["models"][report["primary_action"]]["lm"]["fixed_0_5"]["f1"],
        "primary_passive_lm_f1":passive_result["models"][report["primary_passive"]]["lm"]["fixed_0_5"]["f1"]},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
