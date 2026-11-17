#!/usr/bin/env python3
"""Offline causal/action validation for vvrl.v1 JSONL after real encode."""
from collections import Counter
from pathlib import Path
import argparse
import json
import math
from state_schema import parse_observation


def validate(path):
    records=[]
    for n,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():continue
        try:obj=json.loads(line)
        except json.JSONDecodeError as e:raise ValueError(f'JSON line {n}: {e}') from e
        records.append(obj)
    if not records:raise ValueError('No events')
    seen_feedback=set()
    observations={}
    actions={}
    counts=Counter()
    for n,evt in enumerate(records,1):
        kind=evt.get('kind')
        counts[kind]+=1
        poc=evt.get('poc')
        if kind=='feedback':
            seen_feedback.add(poc)
            if int(evt.get('actual_bits',0))<0:raise ValueError('negative coded bits')
        elif kind=='observation':
            obj,arr=parse_observation(evt)
            if poc in observations:raise ValueError(f'Duplicate observation for POC {poc}')
            if obj['prev_valid'] and obj['prev_feedback_poc'] not in seen_feedback:
                raise ValueError(f'Information leakage: POC={poc} references future feedback')
            observations[poc]=obj
        elif kind=='picture_action':
            if poc not in observations:raise ValueError('action before observation')
            if poc in actions:raise ValueError('duplicate picture action')
            qp0=int(evt['qp_native']);qr=int(evt['qp_requested']);qa=int(evt['qp_accepted'])
            obs=observations[poc]
            if qp0!=obs['native_qp']:raise ValueError('Native QP mismatch')
            if not obs['qp_min']<=qa<=obs['qp_max']:raise ValueError('QP out of bounds')
            l0=float(evt['lambda_native']);la=float(evt['lambda_accepted'])
            if not math.isfinite(l0) or not math.isfinite(la) or l0<=0 or la<=0:
                raise ValueError('Invalid lambda')
            expected=l0*2**((qa-qp0)/3)
            if not math.isclose(expected,la,rel_tol=1e-7,abs_tol=1e-9):
                raise ValueError(f'QP/lambda mismatch POC={poc}')
            actions[poc]=evt
        elif kind=='ctu_map_action':
            if poc not in observations:raise ValueError('Map action before frame observation')
            obs=observations[poc]
            ctu=int(evt['ctu'])
            if not 0<=ctu<obs['ctu_columns']*obs['ctu_rows']:
                raise ValueError('Invalid CTU map index')
        elif kind=='coding_ctu_qp':
            if poc not in observations:raise ValueError('Committed CTU log before frame observation')
        elif kind in ('slice_qp','coding_qp'):
            if poc not in observations:raise ValueError(f'{kind} without observation')
        else:
            raise ValueError(f'Unknown event kind on line {n}: {kind}')
    if set(actions)!=set(observations):
        raise ValueError('Missing action for some observations')
    result={'valid':True,'events':dict(counts),'pictures':len(observations),
            'picture_actions_applied':sum(e['qp_native']!=e['qp_accepted'] for e in actions.values()),
            'coding_ctu_qp_events':counts['coding_ctu_qp']}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('jsonl')
    args=parser.parse_args()
    print(json.dumps(validate(args.jsonl),indent=2))
