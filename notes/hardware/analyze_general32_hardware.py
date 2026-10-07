#!/usr/bin/env python3
"""Offline evidence inventory. Never creates DDS endpoints or robot commands.

Requires the site's existing numpy/mujoco environment. Raw logs and FK CSVs stay
in --artifacts; only the compact JSON/CSV outputs belong in Git.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def numeric(path):
    with path.open() as f:
        header = next(csv.reader(f))
    a = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    return header, a


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'count': 0}
    return dict(count=int(a.size), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(a.max()))


def sample_info(a):
    dt = np.diff(a[:, 3]) / 1000
    good = dt[dt > 0]
    return {'rows': len(a), 'finite': bool(np.isfinite(a).all()),
            'median_rate_hz': float(1 / np.median(good)) if good.size else None,
            'dt_s': stats(good), 'nonpositive_dt_count': int((dt <= 0).sum()),
            'index_gaps': int((np.diff(a[:, 0]) != 1).sum())}


def parameters(path):
    text = re.sub(r'//[^\n]*', '', path.read_text())
    constants = {}
    # Trusted local C++ header; restrict evaluation to numeric expressions/constants.
    for name, expr in re.findall(r'const double (\w+)\s*=\s*([^;]+);', text):
        constants[name] = eval(expr, {'__builtins__': {}}, constants)
    out = {}
    for name in ['isaaclab_to_mujoco', 'mujoco_to_isaaclab', 'g1_action_scale', 'default_angles', 'kps', 'kds']:
        body = re.search(r'\b' + name + r'\s*=\s*\{(.*?)\};', text, re.S).group(1)
        out[name] = [eval(x.strip(), {'__builtins__': {}}, constants) for x in body.split(',') if x.strip()]
        assert len(out[name]) == 29
    assert sorted(out['isaaclab_to_mujoco']) == list(range(29))
    assert all(out['mujoco_to_isaaclab'][v] == i for i, v in enumerate(out['isaaclab_to_mujoco']))
    out['header_path'] = str(path)
    out['header_sha256'] = sha(path)
    return out


def targets(path, p):
    counts, reasons, stages = Counter(), Counter(), Counter()
    first_stop = None
    control_seen = False
    maxdrop = 0
    writer0 = []
    pd_seen = {}
    examples = []
    total = 0
    lows = [float(np.float32(p['default_angles'][i] - 3.00001 * p['g1_action_scale'][i])) for i in range(29)]
    highs = [float(np.float32(p['default_angles'][i] + 3.00001 * p['g1_action_scale'][i])) for i in range(29)]
    pd = [(float(np.float32(p['kps'][i])), float(np.float32(p['kds'][i]))) for i in range(29)]
    with path.open() as f:
        rows = csv.reader(f)
        next(rows)
        for r in rows:
            total += 1
            stage, sent, fault, i = r[1], r[2] == '1', r[3] == '1', int(r[4])
            vals = tuple(map(float, r[6:11]))
            finite = all(map(math.isfinite, vals))
            damping = stage == 'DAMPING'
            maxdrop = max(maxdrop, int(r[14]))
            if r[13]:
                reasons[r[13]] += 1
            if fault:
                counts['fault_rows'] += 1
            if not sent:
                continue
            counts['writer_joint_rows'] += 1
            if i == 0:
                stages[stage] += 1
                writer0.append(int(r[15]))
                if stage.startswith('CONTROL'):
                    control_seen = True
                if damping and control_seen and first_stop is None:
                    first_stop = int(r[15])
                if not damping and first_stop is not None:
                    counts['active_commands_after_control_stop_damping'] += 1
            if damping:
                if vals != (0., 0., 0., 0., 8.):
                    counts['nonexact_damping_joint_rows'] += 1
                continue
            invalid = not finite or vals[3] < 0 or vals[4] < 0 or fault
            if stage.startswith('CONTROL'):
                pd_seen.setdefault(str(i), vals[3:5])
                if vals[3:5] != pd[i]:
                    counts['control_pd_mismatch_joint_rows'] += 1
                # Float32 mapping uses the installed control envelope, not XML.
                if not lows[i] <= vals[0] <= highs[i]:
                    counts['outside_decoder_mapping_envelope_joint_rows'] += 1
            if invalid:
                counts['invalid_active_writer_joint_rows'] += 1
                if len(examples) < 3:
                    examples.append(r)
    intervals = np.diff(writer0) / 1e9
    return {'rows': total, 'writer_stage_command_counts': dict(stages),
            'counts': dict(counts), 'reasons': dict(reasons), 'max_dropped_rows': maxdrop,
            'writer_interval_s': stats(intervals), 'control_pd_seen_hardware_order': pd_seen,
            'first_post_control_damping_steady_ns': first_stop, 'invalid_examples': examples,
            'scope': 'Logged sent=1 writer attempts only; not independent DDS delivery acknowledgement. Dropped diagnostics limit completeness.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-root', type=Path, default=Path('/home/user/code'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--artifacts', type=Path, required=True)
    args = ap.parse_args()
    root, out, artifacts = args.code_root.resolve(), args.output.resolve(), args.artifacts.resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = root / 'ge/SoftSONIC_GuardedBundle_20261006'
    site = root / 'ge/SoftSONIC-controller-guard-v1'
    accept = root / 'ge/guard_acceptance'
    include = site / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include'
    src = include.parent / 'src'
    p = parameters(include / 'policy_parameters.hpp')
    with (accept / 'limits_hardware_user_confirmed_v1.csv').open() as f:
        lines = f.readlines()
    joints = list(csv.DictReader(lines[8:]))
    assert [int(j['index']) for j in joints] == list(range(29))
    p['joints_hardware_PR_order'] = joints
    scene = root / 'ge/SONIC-local-fallback/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    addresses = [int(model.joint(j['name']).qposadr[0]) for j in joints]
    assert len(set(addresses)) == 29
    pelvis = model.body('pelvis').id
    wrists = [model.body(s + '_wrist_yaw_link').id for s in ['left', 'right']]
    models = {name: sha(bundle / 'assets' / name) for name in [
        'general32_native42_residual.onnx', 'sonic_original_g1_encoder.onnx',
        'sonic_original_decoder.onnx', 'observation_config.yaml']}
    assert models['general32_native42_residual.onnx'] == 'f9ad4973f3238db1caf56a821c90fa4ce44b01aedcb9b6f8575393fea23fd107'
    profiles = json.loads((accept / 'general32_profiles.json').read_text())['profiles']
    original_refs = json.loads((bundle / 'assets/references/reference_manifest.json').read_text())
    records = []
    for d in sorted((bundle / 'logs').iterdir()):
        identity_file = d / 'run_identity.json'
        if not identity_file.exists():
            continue
        ident = json.loads(identity_file.read_text())
        if not ident.get('profile', '').startswith('general32') or ident.get('sim') is not False:
            continue
        print('Analyzing', d.name, flush=True)
        meta = ident['profile_metadata']
        profile = ident['profile']
        ref = meta['reference']
        refroot = Path(ref) if Path(ref).is_absolute() else bundle / 'assets' / ref
        refdir, = [x for x in refroot.iterdir() if x.is_dir()]
        actual_reference = {}
        for rel, expected in meta['reference_files_sha256'].items():
            base = Path(meta.get('reference_base', bundle))
            file = base / rel
            actual_reference[rel] = {'sha256': sha(file), 'expected_sha256': expected}
        assert all(v['sha256'] == v['expected_sha256'] for v in actual_reference.values())
        frames = meta.get('frames') or original_refs[refdir.name]['frames']
        source = meta.get('source_key') or original_refs[refdir.name]['source']['source']
        split = meta.get('training_split', 'seen')
        if profile == 'general32_stand':
            source = 'big_light_two_hands_hold_R_001__A508'
            split = 'seen'
        evidence = []
        before = {f.name: (f.stat().st_size, f.stat().st_mtime_ns) for f in d.iterdir() if f.is_file()}
        for f in sorted(d.iterdir()):
            if f.is_file():
                evidence.append({'path': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)})
        qh, q = numeric(d / 'q.csv')
        fields = {}
        for name in ['q', 'dq', 'action', 'base_quat', 'base_ang_vel', 'base_accel', 'torso_quat',
                     'torso_ang_vel', 'torso_accel', 'motor_torque', 'motor_error', 'token_state', 'motion_playing']:
            h, a = numeric(d / (name + '.csv'))
            fields[name] = sample_info(a)
            if name == 'motor_error':
                fields[name]['nonzero_entries'] = int((a[:, 5:] != 0).sum())
            if name == 'action':
                fields[name]['max_abs_raw_action_previous_tick'] = float(np.abs(a[:, 5:]).max())
            if name == 'base_quat':
                fields[name]['quaternion_norm'] = stats(np.linalg.norm(a[:, 5:9], axis=1))
                unit = a[:, 5:9] / np.linalg.norm(a[:, 5:9], axis=1)[:, None]
                fields[name]['imu_up_axis_tilt_deg_max'] = float(np.rad2deg(np.arccos(np.clip(1 - 2*(unit[:, 1]**2 + unit[:, 2]**2), -1, 1))).max())
                fields[name]['tilt_scope'] = 'IMU attitude diagnostic; not a calibrated fall verdict or world-root estimate'
        _, playing = numeric(d / 'motion_playing.csv')
        flags = playing[:, 5] == 1
        trueidx = np.flatnonzero(flags)
        spans = []
        for group in np.split(trueidx, np.flatnonzero(np.diff(trueidx) != 1) + 1) if trueidx.size else []:
            if group.size:
                spans.append({'first_index': int(playing[group[0], 0]), 'last_index': int(playing[group[-1], 0]),
                              'ticks': len(group), 'first_wall_ms': float(playing[group[0], 2]),
                              'last_wall_ms': float(playing[group[-1], 2]),
                              'timestamp_span_s': float((playing[group[-1], 3] - playing[group[0], 3]) / 1000)})
        _, loop = numeric(d / 'control_loop.csv')
        control = loop[loop[:, 4] == 1]
        timing = stats(control[:, 2] / 1000)
        timing['over_20ms_count'] = int((control[:, 2] > 20000).sum())
        timing['incomplete_rows'] = int((loop[:, 4] != 1).sum())
        timing['start_interval_ms'] = stats(loop[loop[:, 3] > 0, 3] / 1000)
        # Residual mode is a string; parse it with DictReader below.
        target = targets(d / 'targets_direct.csv', p)
        startup = [json.loads(x) for x in (d / 'startup_transition.jsonl').read_text().splitlines()]
        stage_frames = [{'stage': x['stage'], 'frame': x['reference_frame'], 'play': x['play'],
                         'steady_ns': x['steady_ns']} for x in startup]
        fk = []
        for r in q:
            data.qpos[:] = model.qpos0
            data.qpos[addresses] = r[5:34]
            mujoco.mj_forward(model, data)
            rotation = data.xmat[pelvis].reshape(3, 3)
            pos = np.concatenate([rotation.T @ (data.xpos[b] - data.xpos[pelvis]) for b in wrists])
            fk.append(np.r_[r[:5], pos])
        fk = np.asarray(fk)
        fkpath = artifacts / (d.name + '_pelvis_wrist_fk.csv')
        np.savetxt(fkpath, fk, delimiter=',', header='index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,left_x_m,left_y_m,left_z_m,right_x_m,right_y_m,right_z_m', comments='', fmt='%.9f')
        fkmetrics = {}
        mask = np.isin(q[:, 0], playing[flags, 0])
        for i, side in enumerate(['left', 'right']):
            v = fk[mask, 5 + 3*i:8 + 3*i]
            if v.size:
                fkmetrics[side] = {'playing_axis_min_m': v.min(axis=0).tolist(), 'playing_axis_max_m': v.max(axis=0).tolist(),
                                   'playing_axis_range_m': np.ptp(v, axis=0).tolist()}
        after = {f.name: (f.stat().st_size, f.stat().st_mtime_ns) for f in d.iterdir() if f.is_file()}
        record = {'run_id': d.name, 'profile': profile, 'mode': ident['mode'], 'gain': ident['gain'],
                  'sim': False, 'identity': ident, 'identity_sha256': sha(identity_file),
                  'controller_dirty_at_run': 'unknown; not captured in run identity',
                  'reference': {'exact_source': source, 'training_split': split, 'frames': frames, 'fps': 50,
                                'duration_ticks_s': frames/50, 'root': str(refroot), 'files': actual_reference,
                                'historic_stand_label_correction': profile == 'general32_stand'},
                  'logs_unchanged_during_analysis': before == after,
                  'sampling_and_finiteness': fields, 'playback': {'true_ticks': int(flags.sum()), 'spans': spans,
                      'full_frame_counts_observed': bool(any(s['ticks'] >= frames for s in spans)),
                      'scope': 'Controller play-state progression; not an independent visual task-success verdict'},
                  'control_ms': timing, 'targets': target, 'startup_stage_samples': stage_frames,
                  'fk': {'scope': 'Official-model pelvis-relative wrist FK estimate from actual SDK q, not world position or force-caused displacement',
                         'csv': str(fkpath), 'sha256': sha(fkpath), 'playing_summary': fkmetrics},
                  'manual_events': {'hands': 'unknown', 'direction': 'unknown', 'force_N': 'unknown',
                      'force_intervals': 'unknown', 'release_times': 'unknown', 'phase_at_manual_pull': 'unknown',
                      'operator': 'onsite operators; individual identity unknown', 'video': 'unknown/not found locally'},
                  'observations': {'playable': 'controller progression only', 'hand_pull_retreat': 'unknown per-run',
                      'release_recovery': 'unknown', 'unforced_hold': 'unknown', 'physical_fall': 'unknown'},
                  'evidence': evidence}
        record['playback']['last_frame_in_startup_trace'] = any(
            s['stage'].startswith('CONTROL') and s['frame'] == frames - 1 for s in stage_frames)
        record['playback']['post_play_logged_hold_ticks'] = int(len(playing) - trueidx[-1] - 1) if trueidx.size else 0
        with (d / 'residual.csv').open() as f:
            rr = list(csv.DictReader(f))
        record['residual_sparse'] = {'rows': len(rr), 'modes': sorted(set(r['mode'] for r in rr)),
            'gains': sorted(set(float(r['gain']) for r in rr)), 'delta_l2': stats([float(r['delta_l2']) for r in rr]),
            'delta_max': stats([float(r['delta_max']) for r in rr]),
            'max_at_clamp_logged_rows': sum(float(r['delta_max']) >= .5 for r in rr),
            'nonzero_norm_logged_rows': sum(float(r['delta_l2']) > 0 for r in rr),
            'inference_ms': stats([float(r['inference_ms']) for r in rr]),
            'scope': 'Sparse norms/maxima (~1Hz plus warmup); full delta64 and component clamp counts not logged'}
        records.append(record)
        save(artifacts / (d.name + '_analysis.json'), record)
    commit = subprocess.check_output(['git', '-C', str(site), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(site), 'diff', '--binary'], text=True)
    status = subprocess.check_output(['git', '-C', str(site), 'status', '--porcelain'], text=True)
    (artifacts / 'controller_dirty_current.patch').write_text(dirty)
    snapshot = {'schema': 'general32-site-identity-v1', 'captured_utc': datetime.now(timezone.utc).isoformat(),
                'controller_local_commit': commit, 'controller_status_porcelain': status,
                'controller_current_dirty_diff_sha256': sha(artifacts / 'controller_dirty_current.patch'),
                'controller_binary_sha256': sha(site / 'gear_sonic_deploy/target/release/g1_deploy_softsonic'),
                'models': models, 'parameters': p, 'profiles_snapshot': profiles,
                'source_sha256': {str(x.relative_to(site)): sha(x) for x in [src/'g1_deploy_softsonic.cpp', src/'state_logger.cpp', include/'softsonic_target_guard.hpp', include/'policy_parameters.hpp']},
                'fk_model': {'scene': str(scene), 'scene_sha256': sha(scene),
                             'compiled_model_nq': model.nq, 'sdk29_joint_qpos_addresses': addresses,
                             'xml_sha256': {str(x): sha(x) for x in sorted(scene.parent.glob('*.xml'))}},
                'joint_limits_sha256': sha(accept / 'limits_hardware_user_confirmed_v1.csv'),
                'native_receipt_sha256': sha(accept / 'general32_native_actor_receipt.json'),
                'export_evidence_sha256': {name: sha(bundle/'assets'/name) for name in
                    ['general_source_export_validation.json', 'general_rms_identity.json', 'golden_inputs_outputs.npz']}}
    save(out / 'general32_hardware_identity_20261007.json', snapshot)
    save(out / 'general32_hardware_experiments_20261007.json', {'schema': 'general32-hardware-evidence-v1',
         'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_count': len(records), 'runs': records,
         'prepared_profiles_without_hardware_run': sorted(set(profiles) - {r['profile'] for r in records}),
         'scope': 'Existing hardware logs only; no new DDS/control processes or experiments launched'})
    columns = ['run_id', 'profile', 'mode', 'source', 'training_split', 'frames', 'playing_ticks', 'control_p99_ms',
               'control_max_ms', 'over20ms', 'fault_rows', 'invalid_active_writer_rows', 'dropped_target_rows',
               'retreat', 'release_recovery', 'hand_direction_force_times', 'evidence_root']
    with (out / 'general32_hardware_experiments_20261007.csv').open('w') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(columns)
        for r in records:
            w.writerow([r['run_id'], r['profile'], r['mode'], r['reference']['exact_source'], r['reference']['training_split'],
                        r['reference']['frames'], r['playback']['true_ticks'], r['control_ms'].get('p99'), r['control_ms'].get('max'),
                        r['control_ms']['over_20ms_count'], r['targets']['counts'].get('fault_rows', 0),
                        r['targets']['counts'].get('invalid_active_writer_joint_rows', 0), r['targets']['max_dropped_rows'],
                        r['observations']['hand_pull_retreat'], 'unknown', 'unknown', str(bundle/'logs'/r['run_id'])])
    print('Completed', len(records), 'hardware runs', flush=True)


if __name__ == '__main__':
    main()
