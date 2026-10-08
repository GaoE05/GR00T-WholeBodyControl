#!/usr/bin/env python3
"""Small fixed-observation CPU replay of existing logs. No DDS/GPU/robot I/O.

Run in the existing guarded bundle venv. Assets stay read-only. A portable rerun
only needs samples.npz, identity.json and the same two ONNX files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path('/home/user/code')
REPO = ROOT/'ge/SoftSONIC-hardware-results-20261007/notes/hardware'
BUNDLE = ROOT/'ge/SoftSONIC_GuardedBundle_20261006'
RUN_IDS = ['20261007_174302_general32_lift_walk_start_A201_on',
           '20261007_174626_general32_high_watering_walk_A501_on']
ACTOR_SHA = 'f9ad4973f3238db1caf56a821c90fa4ce44b01aedcb9b6f8575393fea23fd107'
DECODER_SHA = '7e27a100cd540ea6037d2708ff87a5e2d97b79246d4c658fb3c0522b56b37a6d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(path):
    return np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)


def errors(a, b):
    e = np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))
    return {'max_abs': float(e.max()), 'mean_abs': float(e.mean()),
            'rms': float(np.sqrt(np.mean(e*e)))}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    result = ort.InferenceSession(str(path), sess_options=options,
                                 providers=['CPUExecutionProvider'])
    assert result.get_providers() == ['CPUExecutionProvider']
    return result


def gravity(quat):
    # Exact math_utils.hpp convention: conjugate wxyz; DO NOT normalize.
    inv = quat.copy()
    inv[:, 1:] *= -1
    qw, vec = inv[:, :1], inv[:, 1:]
    v = np.tile([0., 0., -1.], (len(quat), 1))
    return (v*(2*qw*qw-1)+(np.cross(vec, v)*qw)*2
            +(vec*np.sum(vec*v, axis=1, keepdims=True))*2)


def reconstruct(arrays, i, isaac_to_hw, defaults):
    assert i >= 9  # no padding or invented pre-control history in this study
    idx = np.arange(i-9, i+1)
    qisaac = arrays['q'][idx, 5:][:, isaac_to_hw]-defaults[isaac_to_hw]
    result = np.concatenate([
        arrays['base_ang_vel'][idx, 5:].ravel(),
        qisaac.ravel(),
        arrays['dq'][idx, 5:][:, isaac_to_hw].ravel(),
        arrays['action'][idx, 5:].ravel(),
        gravity(arrays['base_quat'][idx, 5:]).ravel()]).astype(np.float32)
    assert result.shape == (930,) and np.isfinite(result).all()
    # CSV current action is previous output, retained exactly for history.
    return result, idx


def mapped(raw_isaac, hw_to_isaac, scales, defaults):
    return (defaults+raw_isaac[hw_to_isaac].astype(np.float64)*scales).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=Path, help='Portable CPU rerun of saved sample arrays')
    ap.add_argument('--identity', type=Path, help='Identity file for portable rerun')
    ap.add_argument('--actor', type=Path, default=BUNDLE/'assets/general32_native42_residual.onnx')
    ap.add_argument('--decoder', type=Path, default=BUNDLE/'assets/sonic_original_decoder.onnx')
    args = ap.parse_args()
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=True)
    assert sha(args.actor) == ACTOR_SHA and sha(args.decoder) == DECODER_SHA
    actor, decoder = session(args.actor), session(args.decoder)
    assert actor.get_inputs()[0].shape == [1, 930]
    assert decoder.get_inputs()[0].shape == [1, 994]
    if args.samples:
        identity = json.loads(args.identity.read_text())
        packed = dict(np.load(args.samples, allow_pickle=False))
    else:
        original = json.loads((REPO/'general32_hardware_identity_20261007.json').read_text())
        inventory = json.loads((REPO/'general32_hardware_experiments_20261007.json').read_text())
        phase_report = json.loads((REPO/'general32_phase_analysis_20261007.json').read_text())
        p = original['parameters']
        isaac_to_hw = np.array(p['mujoco_to_isaaclab'], dtype=int)
        hw_to_isaac = np.array(p['isaaclab_to_mujoco'], dtype=int)
        defaults = np.array(p['default_angles']);scales = np.array(p['g1_action_scale'])
        source, samples, captures, anchor_audits = {}, [], [], []
        for run_id in RUN_IDS:
            run = next(r for r in inventory['runs'] if r['run_id'] == run_id)
            detail = next(r for r in phase_report['priority_details'] if r['run_id'] == run_id)
            log = BUNDLE/'logs'/run_id
            names = ['q','dq','base_ang_vel','base_quat','action','token_state','motion_playing']
            arrays = {n: load(log/(n+'.csv')) for n in names}
            used = [n+'.csv' for n in names]+['residual.csv','startup_transition.jsonl','run_identity.json']
            prior = {Path(e['path']).name: e for e in run['evidence']}
            for name in used:
                file = log/name;digest = sha(file)
                assert digest == prior[name]['sha256'], str(file)
                source[str(file)] = {'sha256': digest, 'bytes': file.stat().st_size}
            for a in arrays.values():
                assert np.array_equal(a[:, :5], arrays['q'][:, :5])
                assert np.array_equal(a[:, 0], np.arange(len(a)))
            aligned_path = Path(detail['aligned_artifact']['path'])
            assert sha(aligned_path) == detail['aligned_artifact']['sha256']
            source[str(aligned_path)] = {'sha256': sha(aligned_path), 'bytes': aligned_path.stat().st_size}
            aligned = np.load(aligned_path, allow_pickle=False)
            phase = aligned['phase'];frame = aligned['nominal_frame']
            n = len(phase)
            # Validate the runtime warmup ContextGate against every sparse diagnostic.
            flag = arrays['motion_playing'][:, 5].astype(bool)
            gate = np.ones(n, dtype=int)
            for i in range(1, n):
                reset = flag[i] != flag[i-1] or frame[i] < frame[i-1] or frame[i] > frame[i-1]+1
                gate[i] = 1 if reset else gate[i-1]+1
            residual = {int(r['tick']): r for r in csv.DictReader((log/'residual.csv').open())}
            for tick, r in residual.items():
                assert gate[tick-1] == int(r['warmup_samples']), (run_id, tick)
                assert (gate[tick-1] >= 10) == (float(r['inference_ms']) > 0)
            snapshot = {int(x['control_tick']): x for x in
                        map(json.loads, (log/'startup_transition.jsonl').read_text().splitlines())
                        if x['raw_valid']}
            for tick, x in snapshot.items():
                if tick <= 10:
                    continue
                i = tick-1
                obs, idx = reconstruct(arrays, i, isaac_to_hw, defaults)
                direct = np.array(x['policy_input_float32'], dtype=np.float32)
                nom = arrays['token_state'][i, 5:].astype(np.float32)
                assert np.array_equal(nom, np.array(x['nominal_token'], dtype=np.float32))
                raw = np.array(x['raw_action_isaaclab'], dtype=np.float32)
                # Direct snapshot and next action CSV share the same float32 output.
                csv_raw = arrays['action'][i+1, 5:].astype(np.float32)
                assert errors(raw, csv_raw)['max_abs'] <= 5.1e-10
                anchor_audits.append({'run_id': run_id, 'tick': tick, 'stage': x['stage'],
                    'phase': str(phase[i]), 'reconstructed_obs_vs_direct_float32': errors(obs, direct[64:]),
                    'nominal_token_vs_snapshot': errors(nom, x['nominal_token']),
                    'terms_max_error': {term: errors(obs[lo:hi], direct[64+lo:64+hi])['max_abs'] for term, lo, hi in
                        [('base_ang_vel',0,30),('joint_pos',30,320),('joint_vel',320,610),('last_actions',610,900),('gravity',900,930)]}})
                captures.append((run_id, tick, obs, nom, direct, raw, gate[i] >= 10))
            # 3 selected ticks per phase: endpoint, interior, exact/sparse anchor.
            selected = {}
            for stage in ['PLAY','FINAL_HOLD']:
                eligible = np.flatnonzero((phase == stage) & (gate >= 10) &
                    np.isfinite(aligned['raw_current_hardware_order']).all(axis=1))
                center = int(eligible[len(eligible)//2])
                for hit, label in [(True,'right_wrist_pitch_endpoint'),(False,'right_wrist_pitch_interior')]:
                    candidates = eligible[(np.abs(aligned['raw_current_hardware_order'][eligible, 27]) == 3) == hit]
                    assert candidates.size, (run_id, stage, label)
                    chosen = int(candidates[np.argmin(np.abs(candidates-center))])
                    selected.setdefault(chosen, []).append(label)
                if stage == 'PLAY':
                    chosen = int(np.flatnonzero(phase == stage)[-1])
                    assert chosen+1 in snapshot and gate[chosen] >= 10
                    selected.setdefault(chosen, []).append('direct_policy_snapshot_last_play_frame')
                else:
                    chosen = 1150-1
                    assert phase[chosen] == stage and gate[chosen] >= 10 and 1150 in residual
                    selected.setdefault(chosen, []).append('sparse_residual_tick_1150')
            for i, reasons in sorted(selected.items()):
                obs, idx = reconstruct(arrays, i, isaac_to_hw, defaults)
                snap = snapshot.get(i+1)
                direct = np.array(snap['policy_input_float32'], np.float32) if snap else np.full(994,np.nan,np.float32)
                reconstructed_obs = obs.copy()
                if snap:
                    obs = direct[64:].copy()  # untouched raw suffix: exact original float32
                samples.append({'run_id': run_id, 'i': i, 'tick': i+1, 'phase': str(phase[i]),
                    'frame': int(frame[i]), 'reasons': reasons, 'obs': obs, 'obs_reconstructed': reconstructed_obs, 'nom': arrays['token_state'][i,5:].astype(np.float32),
                    'indices': idx, 'history_q_hardware': arrays['q'][idx,5:],
                    'history_dq_hardware': arrays['dq'][idx,5:], 'history_base_ang_vel': arrays['base_ang_vel'][idx,5:],
                    'history_base_quat_wxyz': arrays['base_quat'][idx,5:], 'history_previous_raw_isaac': arrays['action'][idx,5:],
                    'recorded_raw_isaac': arrays['action'][i+1,5:].astype(np.float32),
                    'recorded_new_target_hardware': aligned['new_command_q_dq_tau_kp_kd'][i,:,0],
                    'held_command': aligned['held_command_q_dq_tau_kp_kd'][i],
                    'actual_q': aligned['actual_q'][i], 'actual_dq': aligned['actual_dq'][i],
                    'tau_est': aligned['motor_tau_est'][i], 'writer_age_ms': float((aligned['steady_ns'][i]-aligned['held_writer_ns'][i])/1e6),
                    'rounding_ambiguous': bool(aligned['rounding_ambiguous'][i]),
                    'steady_ns': int(aligned['steady_ns'][i]), 'wall_ms': float(aligned['wall_ms'][i]),
                    'direct': direct, 'residual': residual.get(i+1), 'gate': int(gate[i])})
        sample_keys = ['run_id','i','tick','phase','frame','obs','obs_reconstructed','nom','indices','history_q_hardware',
            'history_dq_hardware','history_base_ang_vel','history_base_quat_wxyz','history_previous_raw_isaac',
            'recorded_raw_isaac','recorded_new_target_hardware','held_command','actual_q','actual_dq',
            'tau_est','writer_age_ms','rounding_ambiguous','steady_ns','wall_ms','direct','gate']
        packed = {k: np.array([s[k] for s in samples]) for k in sample_keys}
        packed['capture_run_id'] = np.array([c[0] for c in captures])
        packed['capture_tick'] = np.array([c[1] for c in captures])
        packed['capture_obs'] = np.stack([c[2] for c in captures])
        packed['capture_nom'] = np.stack([c[3] for c in captures])
        packed['capture_policy'] = np.stack([c[4] for c in captures])
        packed['capture_raw'] = np.stack([c[5] for c in captures])
        packed['capture_actor_applied'] = np.array([c[6] for c in captures])
        source_files = [args.actor, args.decoder, BUNDLE/'assets/sonic_original_g1_encoder.onnx',
                        BUNDLE/'assets/observation_config.yaml', BUNDLE/'assets/export_core_manifest.json']
        controller = ROOT/'ge/SoftSONIC-controller-guard-v1/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
        source_files += [controller/'src/g1_deploy_softsonic.cpp',controller/'src/state_logger.cpp',
            controller/'include/math_utils.hpp',controller/'include/softsonic_core.hpp',controller/'include/softsonic_residual.hpp',
            controller/'include/policy_parameters.hpp',
            ROOT/'ge/SoftSONIC-controller-guard-v1/gear_sonic_deploy/target/release/g1_deploy_softsonic']
        for file in source_files:
            source[str(file)] = {'sha256': sha(file), 'bytes': file.stat().st_size}
        identity = {'schema':'general32-cpu-counterfactual-input-v1','actor_sha256':ACTOR_SHA,
            'decoder_sha256':DECODER_SHA,'checkpoint_provenance_sha256':'dbdf998a9f2210372db958b2a98f952bd0801e0a42b4b93c03204f74359c7485',
            'controller_commit':'d9d9a7f03d1454bc0358403d5766fcc3f5772d91',
            'binary_sha256':'7170662a5aa864627d4466ee328072be5c5cdab0c58a7ab0a3d77c4d9a3e9af7',
            'parameters':p,'source_files':source,'run_references':{r['run_id']:r['reference'] for r in inventory['runs'] if r['run_id'] in RUN_IDS},
            'anchor_input_audit':anchor_audits,'sample_metadata':[{'run_id':s['run_id'],'tick':s['tick'],
                'selection_reasons':s['reasons'],'sparse_residual':s['residual']} for s in samples],
            'reconstruction':'10 consecutive current-and-previous states, oldest->newest, term-major [angular30,q290,dq290,previous_action290,gravity30]. q hardware-default then SDK->Isaac; gravity uses conjugate wxyz without normalization.',
            'nominal_token_provenance':'token_state.csv logged after encoder, before ResidualEngine.Apply; independently matches direct snapshot nominal_token.',
            'precision':'CSV has9 fractional digits; reconstruction is approximate, not bit-exact original obs. Four direct float32 policy snapshots quantify error. FSQ nominal grid is exactly float32 representable in selected rows.',
            'normalization':'Original actor ONNX embeds frozen checkpoint RMS and normalized-input clip. Raw obs930 provided once; no manual normalization. Decoder receives raw obs930.',
            'preclip_output':'unknown: original decoder exposes only postclip action. No ONNX graph intermediate extractor installed; no deployment graph/weights altered.',
            'unknown':['human force/release events','force magnitude/direction','motor firmware internal timing/torque saturation','unclipped decoder output','exact full float32 obs at nonsnapshot ticks'],
            'scope':'Fixed recorded state, instantaneous target intervention only. This is not an off closed-loop trajectory or compliance-gain test.'}
        np.savez_compressed(out/'samples.npz',**packed)
        save(out/'identity.json',identity)
    p=identity['parameters'];hw_to_isaac=np.array(p['isaaclab_to_mujoco'],dtype=int)
    defaults=np.array(p['default_angles']);scales=np.array(p['g1_action_scale'])
    joint_names=[j['name'] for j in p['joints_hardware_PR_order']]
    def infer_actor(obs):
        return actor.run(None,{'obs':obs[None].astype(np.float32)})[0][0]
    def infer_decoder(token,obs):
        return decoder.run(None,{'obs_dict':np.concatenate([token,obs])[None].astype(np.float32)})[0][0]
    capture_results=[]
    for j in range(len(packed['capture_tick'])):
        obs,nom,direct=packed['capture_obs'][j],packed['capture_nom'][j],packed['capture_policy'][j]
        delta=infer_actor(obs)
        applied=bool(packed['capture_actor_applied'][j])
        fused=(nom.astype(np.float64)+(delta.astype(np.float64) if applied else 0)).astype(np.float32)
        recon=infer_decoder(fused,obs)
        fixed=decoder.run(None,{'obs_dict':direct[None]})[0][0]
        capture_results.append({'run_id':str(packed['capture_run_id'][j]),'tick':int(packed['capture_tick'][j]),
            'runtime_actor_applied':applied,'reconstructed_input_token_vs_direct':errors(fused,direct[:64]),
            'cpu_direct_policy_vs_recorded_TRT_raw':errors(fixed,packed['capture_raw'][j]),
            'cpu_reconstructed_vs_cpu_direct_raw':errors(recon,fixed),
            'cpu_reconstructed_vs_recorded_TRT_raw':errors(recon,packed['capture_raw'][j]),
            'note':('Last PLAY snapshot includes active fused prefix, never used as nominal.' if applied else
                'First PLAY snapshot is warmup: runtime delta=0; full actor was not applied. Nominal and fused prefix coincide.')})
    results=[];joint_rows=[]
    for j in range(len(packed['tick'])):
        obs,nom=packed['obs'][j],packed['nom'][j]
        delta=infer_actor(obs)
        assert np.isfinite(delta).all() and np.all(np.abs(delta)<=.5)
        fused=(nom.astype(np.float64)+delta.astype(np.float64)).astype(np.float32)
        # Both decoder evaluations share the identical, unnormalized obs930.
        a0=infer_decoder(nom,obs);a1=infer_decoder(fused,obs)
        assert np.isfinite(a0).all() and np.isfinite(a1).all()
        q0=mapped(a0,hw_to_isaac,scales,defaults);q1=mapped(a1,hw_to_isaac,scales,defaults)
        assert int(packed['gate'][j])>=10
        run_id,tick=str(packed['run_id'][j]),int(packed['tick'][j])
        meta=next(s for s in identity['sample_metadata'] if s['run_id']==run_id and s['tick']==tick)
        held=packed['held_command'][j]
        held_err=held[:,0]-packed['actual_q'][j]
        pd=held[:,2]+held[:,3]*held_err+held[:,4]*(held[:,1]-packed['actual_dq'][j])
        sparse=meta['sparse_residual']
        delta_compare=None
        if sparse:
            delta_compare={'computed_delta_max':float(abs(delta).max()),'logged_delta_max':float(sparse['delta_max']),
                'max_abs_stat_error':float(abs(abs(delta).max()-float(sparse['delta_max']))),
                'computed_delta_l2':float(np.linalg.norm(delta.astype(float))), 'logged_delta_l2':float(sparse['delta_l2']),
                'l2_abs_stat_error':float(abs(np.linalg.norm(delta.astype(float))-float(sparse['delta_l2']))),
                'scope':'Sparse printed statistics, not a recorded delta64 vector'}
        joints=[]
        for hw,name in enumerate(joint_names):
            isaac=int(hw_to_isaac[hw]);n=float(a0[isaac]);f=float(a1[isaac])
            nb,fb=abs(n)==3,abs(f)==3
            category=('nominal_already_same_boundary' if nb and fb and n==f else
                'residual_flips_boundary' if nb and fb else
                'residual_moves_off_boundary' if nb else
                'residual_moves_to_boundary' if fb else 'interior_target_change')
            row={'run_id':run_id,'tick':tick,'phase':str(packed['phase'][j]),'nominal_frame':int(packed['frame'][j]),
                'hardware_index':hw,'isaaclab_index':isaac,'joint':name,'category':category,
                'nominal_preclip_action':None,'fused_preclip_action':None,
                'nominal_postclip_action':n,'fused_postclip_action':f,'postclip_action_delta':f-n,
                'nominal_q_target_rad':float(q0[hw]),'fused_q_target_rad':float(q1[hw]),'q_target_delta_rad':float(q1[hw]-q0[hw]),
                'recorded_raw_postclip_action':float(packed['recorded_raw_isaac'][j,isaac]),
                'recorded_new_q_target_rad':float(packed['recorded_new_target_hardware'][j,hw]),
                'held_q_target_rad':float(held[hw,0]),'actual_q_rad':float(packed['actual_q'][j,hw]),
                'actual_dq_rad_s':float(packed['actual_dq'][j,hw]),'held_target_minus_actual_rad':float(held_err[hw]),
                'motor_tau_est_Nm':float(packed['tau_est'][j,hw]),'nominal_pd_proxy_Nm':float(pd[hw]),
                'note':'held error describes prior logged command/state; not future counterfactual target tracking or external force'}
            joints.append(row);joint_rows.append(row)
        results.append({'run_id':run_id,'tick':tick,'phase':str(packed['phase'][j]),'nominal_frame':int(packed['frame'][j]),
            'gate_samples':int(packed['gate'][j]),'selection_reasons':meta['selection_reasons'],
            'steady_ns':int(packed['steady_ns'][j]),'wall_ms':float(packed['wall_ms'][j]),
            'input_provenance':('exact recorded float32 raw suffix at snapshot tick' if np.isfinite(packed['direct'][j]).all()
                else 'CSV reconstructed float32; finite decimal precision, no direct original snapshot'),
            'csv_obs_vs_direct_if_available':(errors(packed['obs_reconstructed'][j],packed['direct'][j,64:])
                if np.isfinite(packed['direct'][j]).all() else None),
            'fused_token_vs_recorded_snapshot_if_available':(errors(fused,packed['direct'][j,:64])
                if np.isfinite(packed['direct'][j]).all() else None),
            'replay_raw_cpu_vs_recorded_TRT':errors(a1,packed['recorded_raw_isaac'][j]),
            'replay_q_target_vs_recorded_new_target_rad':errors(q1,packed['recorded_new_target_hardware'][j]),
            'delta64_cpu':delta.tolist(),'fused_token64_cpu':fused.tolist(),
            'delta_max':float(abs(delta).max()),'delta_l2':float(np.linalg.norm(delta.astype(float))),
            'delta_exact_half_components':int((abs(delta)==.5).sum()),'sparse_delta_check':delta_compare,
            'writer_age_ms':float(packed['writer_age_ms'][j]),'writer_timestamp_rounding_ambiguous':bool(packed['rounding_ambiguous'][j]),
            'arm_max_abs_target_delta_rad':{'left':float(np.max(abs(q1[15:22]-q0[15:22]))),
                'right':float(np.max(abs(q1[22:29]-q0[22:29])))},'joints':joints})
    summary={'schema':'general32-fixed-observation-cpu-counterfactual-v1','sample_count':len(results),
        'providers':actor.get_providers(),'onnxruntime_version':ort.__version__,
        'threads':{'intra_op':1,'inter_op':1},'gain':1.0,
        'normalization':identity['normalization'],'preclip_output':identity['preclip_output'],
        'capture_replay_audit':capture_results,'samples':results,
        'unknown':identity['unknown'],'scope':identity['scope'],
        'max_replay_raw_abs_error':max(r['replay_raw_cpu_vs_recorded_TRT']['max_abs'] for r in results),
        'max_replay_q_target_abs_error_rad':max(r['replay_q_target_vs_recorded_new_target_rad']['max_abs'] for r in results),
        'new_hardware_DDS_sim_runs':0,'deployment_changes':0}
    save(out/'results.json',summary)
    with (out/'joint_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(joint_rows[0]),lineterminator='\n');w.writeheader();w.writerows(joint_rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['samples','capture_replay_audit']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
