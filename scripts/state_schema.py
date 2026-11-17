"""Explicit Python-side S0 state contract for VVenC 1.14.0 QP pilot.

No PCA/model-specific normalisation learned from evaluation videos. Output is 11 float32
components for 8 logical groups: [target_bpp, budget_error, qp, frame_one_hot(3),
 temporal_layer, gop_pos, prev_qp, prev_bit_ratio, prev_valid].
"""
import json
import math
import numpy as np

SCHEMA = 'vvrl.v1'
OBS_DIM = 11
FEATURE_NAMES = ('target_bpp', 'budget_error', 'native_qp', 'is_I', 'is_P',
                 'is_B', 'temporal_layer', 'gop_position', 'prev_qp',
                 'prev_bit_ratio', 'prev_valid')


def parse_observation(line_or_dict):
    obj = json.loads(line_or_dict) if isinstance(line_or_dict,str) else dict(line_or_dict)
    if obj.get('schema') != SCHEMA or obj.get('kind') != 'observation':
        raise ValueError('Wrong VVRL protocol/schema')
    qlow,qhigh = int(obj['qp_min']),int(obj['qp_max'])
    if not qlow<qhigh:raise ValueError('Invalid QP range')
    native=int(obj['native_qp'])
    if not qlow<=native<=qhigh: raise ValueError('Native QP outside valid bounds')
    typ=int(obj['frame_type'])
    if typ not in (0,1,2):raise ValueError('Invalid frame type')
    t=int(obj['temporal_layer']);g=int(obj['gop_position']);size=int(obj['gop_size'])
    if t<0 or t>7 or size<1 or not 0<=g<size:raise ValueError('Invalid GOP data')
    ctu_columns=int(obj['ctu_columns']);ctu_rows=int(obj['ctu_rows'])
    if ctu_columns<1 or ctu_rows<1:raise ValueError('Invalid CTU grid')
    values=[obj['target_bpp'],obj['budget_error'],obj['prev_bit_ratio'],
            obj['spatial_activity'],obj['temporal_activity']]
    if not all(math.isfinite(float(x)) for x in values):raise ValueError('Non-finite observation')
    if float(obj['target_bpp'])<=0:raise ValueError('Invalid target_bpp')
    prev_valid=obj['prev_valid']
    if type(prev_valid) is not bool:raise ValueError('prev_valid must be bool')
    if prev_valid and not qlow<=int(obj['prev_qp'])<=qhigh:
        raise ValueError('Previous QP outside valid bounds')
    if 'prev_feedback_poc' not in obj:raise ValueError('Missing temporal feedback POC')
    if not prev_valid and (obj['prev_qp']!=0 or obj['prev_bit_ratio']!=0 or obj['prev_feedback_poc']!=-1):
        raise ValueError('Invalid missing previous history sentinel')
    if type(obj['spatial_valid']) is not bool or type(obj['temporal_valid']) is not bool:
        raise ValueError('Invalid activity masks')
    # State normalisation below is heuristic for smoke testing. Training should fit
    # mean/variance on TRAIN only, freeze them for val/test.
    scale=max(1,qhigh-qlow)
    vec=np.array([
        math.log1p(float(obj['target_bpp'])),
        max(-5.,min(5.,float(obj['budget_error']))),
        (native-qlow)/scale,
        float(typ==0),float(typ==1),float(typ==2),
        t/7.,g/max(1,size-1),
        (int(obj['prev_qp'])-qlow)/scale if prev_valid else 0.,
        min(10.,max(0.,float(obj['prev_bit_ratio']))) if prev_valid else 0.,
        float(prev_valid)
    ],dtype=np.float32)
    if vec.shape != (OBS_DIM,) or not np.isfinite(vec).all():
        raise ValueError('Wrong observation shape/dtype or nonfinite values')
    return obj,vec


def action_from_policy_index(idx):
    if type(idx) is not int or idx not in range(5):
        raise ValueError('Action index must be integer in 0..4')
    return (-2,-1,0,1,2)[idx]
