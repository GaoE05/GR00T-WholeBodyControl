#!/usr/bin/env python3
"""Offline phase/command-response analysis of the existing 36 hardware logs.

Only numpy and file I/O; no DDS, robot, simulator, model inference or GPU calls.
Full aligned arrays stay in --artifacts. Compact summaries belong in Git.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PRIORITY = {'general32_onehand_A508', 'general32_high_watering_walk_A501',
            'general32_lift_walk_start_A201'}
PHASES = ['CONTROL_PREPLAY', 'PLAY', 'FINAL_HOLD', 'FINAL_NONPLAY_UNVERIFIED', 'BETWEEN_PLAY_UNKNOWN']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def numeric(p):
    return np.loadtxt(p, delimiter=',', skiprows=1, ndmin=2)


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'min': float(x.min()), 'max': float(x.max()),
            'mean': float(x.mean()), 'std': float(x.std()),
            'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)),
            'p99': float(np.percentile(x, 99)), 'range': float(np.ptp(x))}


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def phases(playing):
    flag = playing[:, 5] == 1
    true = np.flatnonzero(flag)
    result = np.full(len(flag), 'CONTROL_PREPLAY', dtype='U24')
    if true.size:
        result[true[0]:true[-1]+1] = 'BETWEEN_PLAY_UNKNOWN'
        result[flag] = 'PLAY'
        result[true[-1]+1:] = 'FINAL_HOLD'
    return result


def longest(hit, time_ns, source_index):
    idx = np.flatnonzero(hit)
    if not idx.size:
        return {'ticks': 0, 'tick_duration_s': 0., 'timestamp_span_s': 0.}
    boundary = (np.diff(idx) != 1) | (np.diff(source_index[idx]) != 1)
    groups = np.split(idx, np.flatnonzero(boundary)+1)
    group = max(groups, key=len)
    return {'ticks': len(group), 'tick_duration_s': len(group)*.02,
            'timestamp_span_s': float((time_ns[group[-1]]-time_ns[group[0]])/1e9),
            'first_local_index': int(group[0]), 'last_local_index': int(group[-1])}


def changes(x, ns, source_index):
    good = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.diff(source_index) == 1)
    dt = np.diff(ns)/1e9
    good &= dt > 0
    diff = np.diff(x)[good]
    return {'adjacent_pairs': int(good.sum()), 'total_variation_rad': float(np.abs(diff).sum()),
            'abs_step_rad': stats(np.abs(diff)), 'abs_fd_velocity_rad_s': stats(np.abs(diff/dt[good]))}


def read_targets(path):
    """Group each 29-row trace block; intern repeated writer commands by sequence/stage."""
    vectors, publications, candidates = {}, [], []
    seen = np.zeros(29, dtype=bool)
    matrix = np.zeros((29, 5))
    oldkey = None
    incomplete = inconsistent = drop = 0

    def flush(key):
        nonlocal incomplete, inconsistent
        if key is None:
            return
        if not seen.all():
            incomplete += 1
            return
        ns, seq, stage, sent, fault = key
        vk = (seq, stage)
        if vk in vectors:
            if not np.array_equal(vectors[vk], matrix):
                inconsistent += 1
                return
        else:
            vectors[vk] = matrix.copy()
        (publications if sent else candidates).append((ns, vk, fault))

    with path.open() as f:
        rows = csv.reader(f)
        next(rows)
        for r in rows:
            key = (int(r[15]), int(r[0]), r[1], r[2] == '1', r[3] == '1')
            if key != oldkey:
                flush(oldkey)
                oldkey = key
                seen[:] = False
            i = int(r[4])
            assert not seen[i]
            seen[i] = True
            matrix[i] = list(map(float, r[6:11]))
            drop = max(drop, int(r[14]))
        flush(oldkey)
    publications.sort(key=lambda x: x[0])
    candidates.sort(key=lambda x: x[0])
    return vectors, publications, candidates, {'incomplete_blocks': incomplete,
        'inconsistent_same_sequence_stage_blocks': inconsistent, 'max_dropped_joint_rows': drop}


def align_targets(path, state, loop, raw, parameters):
    vectors, pub, cand, audit = read_targets(path)
    n = len(state)
    starts = loop[:, 1].astype(np.int64)
    ticks = loop[:, 0].astype(np.int64)
    qtick = state[:, 0].astype(np.int64)+1
    assert np.array_equal(ticks[:n], qtick)
    new = np.full((n, 29, 5), np.nan)
    new_ns = np.full(n, -1, dtype=np.int64)
    new_source = np.full(n, 'missing', dtype='U16')
    offsets = []
    for ns, key, fault in cand:
        if not key[1].startswith('CONTROL') or fault:
            continue
        li = int(np.searchsorted(starts, ns, side='right')-1)
        if li < 0 or li >= n:
            continue
        # Candidate timestamps must be inside this completed callback, not merely nearby.
        assert ns <= starts[li]+int(round(loop[li, 2]*1000))
        offsets.append(key[0]-int(ticks[li]))
        new[li] = vectors[key]
        new_ns[li] = ns
        new_source[li] = 'sent0_candidate'
    assert len(set(offsets)) == 1
    offset = offsets[0]
    # Missing diagnostic candidate blocks can be filled from an actually logged publication.
    for ns, key, fault in pub:
        if not key[1].startswith('CONTROL') or fault:
            continue
        tick = key[0]-offset
        i = tick-1
        if 0 <= i < n and new_source[i] == 'missing':
            new[i] = vectors[key]
            new_ns[i] = ns
            new_source[i] = 'writer_fallback'
    pub_ns = np.array([v[0] for v in pub], dtype=np.int64)
    # CSV steady time is printed to 0.001 ms; its rounding uncertainty is ±0.5 us.
    state_ns = np.rint(state[:, 3]*1e6).astype(np.int64)
    pi = np.searchsorted(pub_ns, state_ns, side='right')-1
    assert np.all(pi >= 0)
    held = np.array([vectors[pub[k][1]] for k in pi])
    held_ns = pub_ns[pi]
    ambiguous = np.zeros(n, dtype=bool)
    for i, k in enumerate(pi):
        near = [k] + ([k+1] if k+1 < len(pub) else [])
        ambiguous[i] = any(abs(int(pub_ns[t])-int(state_ns[i])) <= 500 for t in near)
    defaults = np.array(parameters['default_angles'])
    scales = np.array(parameters['g1_action_scale'])
    expected = (defaults+raw*scales).astype(np.float32).astype(float)
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(new[:, :, 0]).all(axis=1)
    shifted_error = np.abs(expected[valid]-new[valid, :, 0])
    # Compare against the erroneous unshifted CSV interpretation as an audit, not a new gate.
    unshifted = np.vstack([np.full((1, 29), np.nan), raw[:-1]])
    uv = np.isfinite(unshifted).all(axis=1) & np.isfinite(new[:, :, 0]).all(axis=1)
    erroneous = (defaults+unshifted[uv]*scales).astype(np.float32).astype(float)
    audit.update({'matched_raw_and_target_ticks': int(valid.sum()),
        'correct_shift_abs_target_error_rad': stats(shifted_error.flatten()),
        'incorrect_unshifted_abs_target_error_rad': stats(np.abs(erroneous-new[uv, :, 0]).flatten()),
        'new_target_sources': dict(Counter(new_source)), 'sequence_minus_control_tick': offset,
        'held_writer_age_ms': stats((state_ns-held_ns)/1e6),
        'state_time_rounding_ambiguous_rows': int(ambiguous.sum()),
        'scope': 'As-of sent=1 trace at logger steady timestamp; not DDS acknowledgement or motor internal command timing. LowState source acquisition age is unlogged.'})
    return new, held, new_ns, held_ns, ambiguous, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', type=Path, required=True)
    ap.add_argument('--artifacts', type=Path, required=True)
    args = ap.parse_args()
    report, artifact = args.report_dir.resolve(), args.artifacts.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    parent = json.loads((report/'general32_hardware_experiments_20261007.json').read_text())
    identity = json.loads((report/'general32_hardware_identity_20261007.json').read_text())
    p = identity['parameters']
    hw_to_isaac = np.array(p['isaaclab_to_mujoco'])
    names = [v['name'] for v in p['joints_hardware_PR_order']]
    overview, detailed, joint_rows, peak_rows = [], [], [], []
    for run in parent['runs']:
        d = Path(run['evidence'][0]['path']).parent
        print('phase scan', d.name, flush=True)
        expected_files = {Path(x['path']).name: x for x in run['evidence']}
        used = ['action.csv', 'q.csv', 'motion_playing.csv', 'control_loop.csv', 'residual.csv', 'startup_transition.jsonl']
        if run['profile'] in PRIORITY:
            used += ['dq.csv', 'motor_torque.csv', 'targets_direct.csv']
        provenance = {}
        before = {}
        for name in used:
            file = d/name
            before[name] = (file.stat().st_size, file.stat().st_mtime_ns)
            digest = sha(file)
            assert digest == expected_files[name]['sha256'], 'Source changed since prior inventory: '+str(file)
            provenance[name] = {'path': str(file), 'sha256': digest, 'bytes': file.stat().st_size}
        state = numeric(d/'q.csv')
        action = numeric(d/'action.csv')
        playing = numeric(d/'motion_playing.csv')
        loop = numeric(d/'control_loop.csv')
        assert np.array_equal(state[:, 0], action[:, 0])
        assert np.array_equal(state[:, 0], playing[:, 0])
        assert np.array_equal(np.diff(state[:, 0]), np.ones(len(state)-1))
        phase = phases(playing)
        play_indices = np.flatnonzero(playing[:, 5] == 1)
        single_complete = (len(play_indices) == run['reference']['frames']
                           and np.all(np.diff(play_indices) == 1)
                           and run['playback']['last_frame_in_startup_trace'])
        if not single_complete:
            phase[phase == 'FINAL_HOLD'] = 'FINAL_NONPLAY_UNVERIFIED'
        # action at CSV index i+1 is produced at CONTROL index i, hardware reorder afterwards.
        raw = np.full((len(state), 29), np.nan)
        raw[:-1] = action[1:, 5:34][:, hw_to_isaac]
        with (d/'residual.csv').open() as f:
            residual = list(csv.DictReader(f))
        startup = [json.loads(x) for x in (d/'startup_transition.jsonl').read_text().splitlines()]
        runsummary = {'run_id': run['run_id'], 'profile': run['profile'], 'mode': run['mode'],
            'gain': run['gain'], 'phases': {}, 'raw_current_last_tick_missing': 1,
            'single_complete_reference_verified': bool(single_complete),
            'manual_force_and_release_events': 'unknown; not inferred', 'source_files': provenance}
        # No continuous state/torque/action log exists before CONTROL.
        runsummary['init_wait'] = {'scope': 'Sparse startup snapshots only; no residual/decoder inference in these states',
            'snapshots': [{k: x[k] for k in ['stage', 'steady_ns', 'actual_q_hardware_rad',
                'actual_dq_hardware_rad_s', 'q_target_float32_hardware_rad', 'kp', 'kd']}
                for x in startup if x['stage'] in ['INIT', 'WAIT', 'CONTROL_ENTER']]}
        for stage in PHASES:
            mask = phase == stage
            if not mask.any():
                continue
            valid = mask & np.isfinite(raw).all(axis=1)
            sample = [r for r in residual if 1 <= int(r['tick']) <= len(state) and phase[int(r['tick'])-1] == stage]
            active = [r for r in sample if float(r['inference_ms']) > 0]
            below = [r for r in active if float(r['delta_max']) < .5
                     and np.isfinite(raw[int(r['tick'])-1]).all()]
            coincidence = {'residual_below_0_5_with_current_raw_available': len(below),
                'raw_endpoint_samples_by_hardware_index': {
                    str(hw): int(sum(abs(raw[int(r['tick'])-1, hw]) == 3 for r in below))
                    for hw in range(29)},
                'scope': 'Same producer tick, sparse inferred residual samples only; not all unsampled ticks'}
            saturation = []
            for hw, name in enumerate(names):
                a = raw[valid, hw]
                hit = np.abs(a) == 3
                if hit.any():
                    ix = np.flatnonzero(valid)
                    first = int(ix[np.flatnonzero(hit)[0]])
                    saturation.append({'hardware_index': hw, 'isaaclab_index': int(hw_to_isaac[hw]),
                        'name': name, 'n': len(a), 'positive_3': int((a == 3).sum()),
                        'negative_3': int((a == -3).sum()), 'clip_fraction': float(hit.mean()),
                        'first_producer_control_tick': first+1, 'first_csv_raw_source_index': first+1,
                        'first_producer_wall_ms': float(state[first, 2]),
                        'longest_clip_any_sign': longest(hit, np.rint(state[valid, 3]*1e6).astype(np.int64), state[valid, 0]),
                        'longest_positive_3': longest(a == 3, np.rint(state[valid, 3]*1e6).astype(np.int64), state[valid, 0]),
                        'longest_negative_3': longest(a == -3, np.rint(state[valid, 3]*1e6).astype(np.int64), state[valid, 0])})
            runsummary['phases'][stage] = {'state_ticks': int(mask.sum()), 'raw_available_ticks': int(valid.sum()),
                'time_span_s': float(np.ptp(state[mask, 3])/1000), 'tick_duration_s': float(mask.sum()*.02),
                'residual': {'logged_samples': len(sample), 'inferred_samples': len(active),
                    'logged_delta_max_equal_0_5': sum(float(r['delta_max']) == .5 for r in sample),
                    'inferred_delta_max_equal_0_5': sum(float(r['delta_max']) == .5 for r in active),
                    'same_tick_residual_below_limit_raw_endpoints': coincidence,
                    'delta_l2': stats([float(r['delta_l2']) for r in sample]),
                    'delta_max': stats([float(r['delta_max']) for r in sample]),
                    'scope': 'Printed sparse delta_max=.5 samples only; not full delta64 component saturation or continuous clamp rate'},
                'raw_saturation': saturation}
        overview.append(runsummary)
        if run['profile'] not in PRIORITY:
            continue
        print('writer/state alignment', d.name, flush=True)
        dq = numeric(d/'dq.csv')
        torque = numeric(d/'motor_torque.csv')
        assert np.array_equal(state[:, :5], dq[:, :5])
        assert np.array_equal(state[:, :5], torque[:, :5])
        new, held, new_ns, held_ns, ambiguous, audit = align_targets(d/'targets_direct.csv', state, loop, raw, p)
        state_ns = np.rint(state[:, 3]*1e6).astype(np.int64)
        actual = state[:, 5:34]
        velocity = dq[:, 5:34]
        tau = torque[:, 5:34]
        held_error = held[:, :, 0]-actual
        pd = held[:, :, 2]+held[:, :, 3]*held_error+held[:, :, 4]*(held[:, :, 1]-velocity)
        true = np.flatnonzero(playing[:, 5] == 1)
        assert len(true) == run['reference']['frames'] and np.all(np.diff(true) == 1)
        frame = np.zeros(len(state), dtype=np.int64)
        frame[true] = np.arange(len(true))
        frame[true[-1]+1:] = len(true)-1
        detail = {'run_id': run['run_id'], 'profile': run['profile'], 'mode': run['mode'], 'gain': run['gain'],
            'alignment_audit': audit, 'joint_phase_metrics': [], 'init_wait': runsummary['init_wait'],
            'manual_events': 'unknown; no force/release event classification',
            'pd_proxy_scope': 'Nominal PD demand from logged published command and SDK snapshot; not firmware measured torque, external force, or a saturation proof'}
        for stage in PHASES:
            mask = (phase == stage) & ~ambiguous
            if not mask.any():
                continue
            ix = np.flatnonzero(mask)
            for hw, name in enumerate(names):
                r = raw[mask, hw]
                h = held[mask, hw, 0]
                q = actual[mask, hw]
                err = held_error[mask, hw]
                clipped = np.abs(r) == 3
                positive = r == 3
                negative = r == -3
                row = {'run_id': run['run_id'], 'profile': run['profile'], 'mode': run['mode'],
                    'phase': stage, 'hardware_index': hw, 'isaaclab_index': int(hw_to_isaac[hw]), 'joint': name,
                    'state_n': len(ix), 'raw_n': int(np.isfinite(r).sum()),
                    'raw_positive_3': int(positive.sum()), 'raw_negative_3': int(negative.sum()),
                    'raw_clip_fraction': float(clipped.sum()/np.isfinite(r).sum()) if np.isfinite(r).any() else None,
                    'new_target_rad': stats(new[mask, hw, 0]), 'held_target_rad': stats(h),
                    'actual_q_rad': stats(q), 'actual_abs_dq_rad_s': stats(np.abs(velocity[mask, hw])),
                    'abs_held_target_error_rad': stats(np.abs(err)),
                    'signed_held_target_error_rad': stats(err),
                    'abs_new_target_minus_actual_now_rad': stats(np.abs(new[mask, hw, 0]-q)),
                    'signed_tau_est_Nm': stats(tau[mask, hw]), 'abs_tau_est_Nm': stats(np.abs(tau[mask, hw])),
                    'nominal_pd_demand_Nm': stats(pd[mask, hw]),
                    'abs_pd_minus_tau_est_Nm': stats(np.abs(pd[mask, hw]-tau[mask, hw])),
                    'pd_tau_est_correlation': corr(pd[mask, hw], tau[mask, hw]),
                    'held_target_actual_position_correlation': corr(h, q),
                    'held_target_changes': changes(h, state_ns[mask], state[mask, 0]),
                    'actual_changes': changes(q, state_ns[mask], state[mask, 0]),
                    'abs_error_at_raw_clip_rad': stats(np.abs(err[clipped])),
                    'abs_error_without_raw_clip_rad': stats(np.abs(err[np.isfinite(r) & ~clipped])),
                    'scope': 'Phase descriptive statistics with unknown human input; not compliance/causal off-on gain.'}
                detail['joint_phase_metrics'].append(row)
                joint_rows.append(row)
                # Three data events, explicitly not human pull/release labels.
                picks = [('largest_held_target_error', int(ix[np.argmax(np.abs(err))]))]
                if np.any(positive):
                    picks.append(('first_raw_plus3', int(ix[np.flatnonzero(positive)[0]])))
                if np.any(negative):
                    picks.append(('first_raw_minus3', int(ix[np.flatnonzero(negative)[0]])))
                for kind, i in picks:
                    peak_rows.append({'run_id': run['run_id'], 'profile': run['profile'], 'mode': run['mode'],
                        'phase': stage, 'event': kind, 'hardware_index': hw, 'isaaclab_index': int(hw_to_isaac[hw]),
                        'joint': name, 'control_tick': i+1, 'state_csv_index': int(state[i, 0]),
                        'raw_source_csv_index': i+1 if i+1 < len(state) else None,
                        'wall_ms': float(state[i, 2]), 'steady_ns': int(state_ns[i]),
                        'nominal_frame_reconstructed': int(frame[i]),
                        'raw_action_current': float(raw[i, hw]) if np.isfinite(raw[i, hw]) else None,
                        'new_q_target_rad': float(new[i, hw, 0]) if np.isfinite(new[i, hw, 0]) else None,
                        'held_q_target_rad': float(held[i, hw, 0]), 'actual_q_rad': float(actual[i, hw]),
                        'actual_dq_rad_s': float(velocity[i, hw]), 'tau_est_Nm': float(tau[i, hw]),
                        'held_kp': float(held[i, hw, 3]), 'held_kd': float(held[i, hw, 4]),
                        'nominal_pd_demand_Nm': float(pd[i, hw]),
                        'writer_age_ms': float((state_ns[i]-held_ns[i])/1e6),
                        'human_event': 'unknown; numeric event only'})
        path = artifact/(run['run_id']+'_aligned.npz')
        np.savez_compressed(path, state_index=state[:, 0], control_tick=state[:, 0]+1,
            steady_ns=state_ns, wall_ms=state[:, 2], phase=phase, nominal_frame=frame,
            raw_current_hardware_order=raw, new_command_q_dq_tau_kp_kd=new,
            new_command_ns=new_ns, held_command_q_dq_tau_kp_kd=held,
            held_writer_ns=held_ns, actual_q=actual, actual_dq=velocity,
            motor_tau_est=tau, nominal_pd_demand=pd, rounding_ambiguous=ambiguous)
        detail['aligned_artifact'] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        for name in used:
            file = d/name
            assert before[name] == (file.stat().st_size, file.stat().st_mtime_ns)
        detailed.append(detail)
        save(artifact/(run['run_id']+'_detail.json'), detail)
    result = {'schema': 'general32-existing-hardware-phase-analysis-v1', 'all_run_count': len(overview),
        'detailed_priority_run_count': len(detailed), 'priority_profiles': sorted(PRIORITY),
        'analysis_scope': 'Offline existing data only; no new robot/simulator/DDS/inference/GPU operations',
        'phase_definition': {'INIT_WAIT': 'Sparse snapshots; no continuous q/dq/torque or decoder/residual observations',
            'CONTROL_PREPLAY': 'Continuous control before first play=true row; startup/warmup included',
            'PLAY': 'motion_playing=true, including the last nominal frame despite CONTROL_HOLD writer label',
            'FINAL_HOLD': 'After a verified complete single reference; not evidence of release or unforced hold',
            'FINAL_NONPLAY_UNVERIFIED': 'Nonplay after last play row without complete single-reference proof; terminal nominal frame unverified',
            'BETWEEN_PLAY_UNKNOWN': 'Nonplay between play intervals, if any; no invented pause semantics'},
        'alignment_method': 'raw CSV row i+1 -> producer control row i; hardware map+scale+default float32; verify against sent0 candidates. Held command is latest logged sent1 at or before state logger timestamp, rounding-ambiguous rows excluded from joint summaries.',
        'identity_snapshot_sha256': sha(report/'general32_hardware_identity_20261007.json'),
        'input_manifest_sha256': sha(report/'general32_hardware_experiments_20261007.json'),
        'overview_36_runs': overview, 'priority_details': detailed,
        'unknown': ['force/release event times', 'force magnitude/direction', 'exact manual phase/hand',
            'external wrench', 'motor firmware commanded/clipped torque', 'unclipped decoder output',
            'continuous delta64/component clipping', 'LowState original acquisition timestamp']}
    # Full descriptive distributions remain onsite; Git carries selected statistics.
    save(artifact/'phase_analysis_full.json', result)
    selected = {'new_target_rad': ['min', 'max', 'std'], 'held_target_rad': ['min', 'max', 'std'],
        'actual_q_rad': ['min', 'max', 'std'], 'actual_abs_dq_rad_s': ['p95', 'max'],
        'abs_held_target_error_rad': ['p50', 'p95', 'max'],
        'abs_new_target_minus_actual_now_rad': ['p50', 'p95', 'max'],
        'signed_tau_est_Nm': ['min', 'max'], 'abs_tau_est_Nm': ['p95', 'max'],
        'nominal_pd_demand_Nm': ['min', 'max'], 'abs_pd_minus_tau_est_Nm': ['p50', 'p95']}
    scalar_metrics = ['run_id', 'profile', 'mode', 'phase', 'hardware_index', 'isaaclab_index',
        'joint', 'state_n', 'raw_n', 'raw_positive_3', 'raw_negative_3', 'raw_clip_fraction',
        'pd_tau_est_correlation', 'held_target_actual_position_correlation']
    result['full_descriptive_artifact'] = {'path': str(artifact/'phase_analysis_full.json'),
                                          'sha256': sha(artifact/'phase_analysis_full.json')}
    for detail in result['priority_details']:
        compact = []
        for row in detail['joint_phase_metrics']:
            c = {k: row[k] for k in scalar_metrics}
            c.update({k: {field: row[k].get(field) for field in fields} for k, fields in selected.items()})
            for k in ['held_target_changes', 'actual_changes']:
                c[k] = {field: row[k][field] for field in ['adjacent_pairs', 'total_variation_rad']}
            compact.append(c)
        detail['joint_phase_metrics'] = compact
    save(report/'general32_phase_analysis_20261007.json', result)
    scalar = ['run_id', 'profile', 'mode', 'phase', 'hardware_index', 'isaaclab_index', 'joint',
              'state_n', 'raw_n', 'raw_positive_3', 'raw_negative_3', 'raw_clip_fraction']
    nested = {'held_target_rad': ['min', 'max', 'std'], 'actual_q_rad': ['min', 'max', 'std'],
              'abs_held_target_error_rad': ['p50', 'p95', 'max'], 'actual_abs_dq_rad_s': ['p95', 'max'],
              'signed_tau_est_Nm': ['min', 'max'], 'abs_tau_est_Nm': ['p95', 'max'],
              'nominal_pd_demand_Nm': ['min', 'max'], 'abs_pd_minus_tau_est_Nm': ['p50', 'p95']}
    columns = scalar+[k+'_'+v for k, values in nested.items() for v in values]
    with (report/'general32_phase_joint_metrics_20261007.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n'); w.writeheader()
        for row in joint_rows:
            flat = {k: row[k] for k in scalar}
            flat.update({k+'_'+v: row[k].get(v) for k, values in nested.items() for v in values})
            w.writerow(flat)
    with (report/'general32_phase_numeric_events_20261007.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(peak_rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(peak_rows)
    print('Completed:', len(overview), 'overview runs,', len(detailed), 'aligned runs,', len(joint_rows), 'joint/phase rows', flush=True)


if __name__ == '__main__':
    main()
