"""Gymnasium 5-return API backed by real full-sequence VVC replays.

CRITICAL: 'replay' is NOT online streaming. Every step re-encodes and decodes
all N frames. Current reward may change if future actions change due to
hierarchical GOP prediction. Do not feed PPO with these trajectories until
this limitation is resolved and true synchronous AU-level interaction exists.
"""
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import time
import shutil
import numpy as np
from scripts.state_schema import action_from_policy_index
from .reward import compose_reward,current_average_bitrate,precision_proxy,InvalidReward
from .runner import FullSequenceReplay,CodecFailure


def _sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym=None;spaces=None


class _ReplayEnv:
    def __init__(self, backend:FullSequenceReplay, detector, *, iou_threshold=.5, reward_lambda=20., episode_id='sequence'):
        self.backend=backend;self.detector=detector
        self.iou_threshold=float(iou_threshold);self.reward_lambda=float(reward_lambda)
        self.episode_id=str(episode_id)
        self.raw_cache={};self.actions={};self.current=None;self.order=[];self.index=0;self.terminated=False
        self.events_file=backend.workdir/'stage4_rewards.jsonl'
        self.episode_counter=0;self.current_episode_id=None

    def _info(self,poc,rollout,perf=None):
        return {'episode_id':self.current_episode_id,'frame_index':poc,'action_order_index':self.index,
                'mode':'real_codec_full_sequence_replay_NOT_online','bitstream':str(rollout.output_266),
                'latency':perf or {},'status':'validated_real_run'}

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('reset options unsupported; use one preprocessed video per instance')
        self.raw_cache={};self.actions={};self.index=0;self.terminated=False
        self.episode_counter+=1;self.current_episode_id=f'{self.episode_id}_{self.episode_counter:04d}'
        self.current=self.backend.run({})
        self.order=[o['poc'] for o,_ in self.current.observations]
        if len(self.order)!=self.backend.n:raise CodecFailure('incomplete initial observations')
        observation=self.current.observations[0][1]
        return observation.copy(),self._info(self.order[0],self.current)

    def step(self,action):
        if self.current is None or self.terminated:raise RuntimeError('step after termination or before reset')
        if isinstance(action,np.integer):action=int(action)
        delta=action_from_policy_index(action)
        poc=self.order[self.index]
        self.actions[poc]=delta
        t0=time.perf_counter()
        previous=self.current
        updated=self.backend.run(self.actions)
        replay_wall=time.perf_counter()-t0
        order=[ob['poc'] for ob,_ in updated.observations]
        if order!=self.order:raise CodecFailure('coding order changed after action: invalid replay transition')
        # State for current step MUST be identical: it was emitted before this action.
        old=self.current.observations[self.index][1]
        if not np.array_equal(old,updated.observations[self.index][1]):
            raise CodecFailure('pre-action state changed after requested action; replay is not causally consistent')
        t_detect=time.perf_counter()
        if poc not in self.raw_cache:
            self.raw_cache[poc]=self.detector.detect(self.backend.original.rgb(poc))
        raw=self.raw_cache[poc]
        raw_detection_seconds=time.perf_counter()-t_detect
        t_rec=time.perf_counter()
        reconstruction=self.detector.detect(updated.reconstructed.rgb(poc))
        reconstruction_detection_seconds=time.perf_counter()-t_rec
        detect_seconds=time.perf_counter()-t_detect
        t_reward=time.perf_counter()
        rate=current_average_bitrate(updated.bits_by_poc,poc,self.backend.settings.fps)
        precision,matched,fp=precision_proxy(raw,reconstruction,self.iou_threshold)
        scores=compose_reward(rate,self.backend.settings.target_bitrate,precision,self.reward_lambda)
        reward_seconds=time.perf_counter()-t_reward
        applied=next(x for x in updated.records if x.get('kind')=='picture_action' and x.get('poc')==poc)
        prior_qp=next(x for x in previous.records if x.get('kind')=='coding_qp' and x.get('poc')==poc)
        after_qp=next(x for x in updated.records if x.get('kind')=='coding_qp' and x.get('poc')==poc)
        prior_bitstream=_sha256_file(previous.output_266)
        after_bitstream=_sha256_file(updated.output_266)
        if delta==0 and prior_bitstream!=after_bitstream:
            raise CodecFailure('identical action plan produced differing VVC bitstreams; replay is non-deterministic')
        if applied['qp_accepted']!=applied['qp_native']:
            if prior_bitstream==after_bitstream:
                raise CodecFailure('nonzero accepted QP action produced unchanged bitstream')
            keys=('qp_min','qp_max','qp_mean_unweighted')
            if all(prior_qp[k]==after_qp[k] for k in keys):
                raise CodecFailure('nonzero accepted QP action produced no detectable coded-CU QP change')
        self.current=updated;self.index+=1
        self.terminated=self.index>=len(self.order)
        next_ob=np.zeros(11,dtype=np.float32) if self.terminated else updated.observations[self.index][1].copy()
        timing={'encode_seconds':updated.encode_seconds,'decode_seconds':updated.decode_seconds,
                'detection_seconds':detect_seconds,'detection_raw_seconds':raw_detection_seconds,
                'detection_reconstructed_seconds':reconstruction_detection_seconds,
                'reward_seconds':reward_seconds,'full_step_seconds':replay_wall+detect_seconds+reward_seconds}
        info=self._info(poc,updated,timing)
        info.update(scores);info.update({'target_bitrate':self.backend.settings.target_bitrate,
            'actual_bitrate':rate,'requested_delta_qp':delta,
            'accepted_delta_qp':applied['qp_accepted']-applied['qp_native'],
            'qp_native':applied['qp_native'],'qp_accepted':applied['qp_accepted'],
            'raw_detections':len(raw),'reconstructed_detections':len(reconstruction),
            'true_positive_matches':matched,'false_positives':fp,
            'raw_boxes':[asdict(b) for b in raw],
            'reconstructed_boxes':[asdict(b) for b in reconstruction],
            'detector_config':getattr(self.detector,'config',{}),
            'encoder_status':'completed','decoder_status':'completed'})
        # Retain jsonl/stderr telemetry but avoid quadratic retention of huge YUV/AU files.
        previous.output_266.unlink(missing_ok=True)
        previous.reconstructed.path.unlink(missing_ok=True)
        with self.events_file.open('a',encoding='utf8') as f:
            f.write(json.dumps(info,allow_nan=False)+'\n')
        return next_ob,float(scores['total_reward']),bool(self.terminated),False,info

    def close(self):
        self.current=None;self.raw_cache.clear()


if gym is not None:
    class VVCReplayEnv(_ReplayEnv,gym.Env):
        """Gym interface for costly smoke validation, not production PPO."""
        metadata={'render_modes':[]}
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(11,),dtype=np.float32)
            self.action_space=spaces.Discrete(5)
        def reset(self,*,seed=None,options=None):
            gym.Env.reset(self,seed=seed)
            return super().reset(seed=seed,options=options)
else:
    class VVCReplayEnv(_ReplayEnv):
        def __init__(self,*args,**kwargs):
            raise ImportError('Install gymnasium to instantiate VVCReplayEnv; reward/unit tests do not need Gymnasium')
