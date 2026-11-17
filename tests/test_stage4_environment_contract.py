"""Mock-backed API/transition unit tests ONLY; NOT real VVenC/VVdeC/YOLO tests.

These tests deliberately never claim codec conformance or genuine detector reward.
They cover Python reset/step/termination/logging without downloading binaries.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from rl_env.environment import _ReplayEnv
from rl_env.reward import Box


class _SyntheticFrame:
    def __init__(self,path):self.path=path
    def rgb(self,poc):return np.zeros((2,4,3),dtype=np.uint8)


class _DeterministicFixtureDetector:
    config={'purpose':'contract_fixture_not_real_YOLO'}
    def detect(self,rgb):return [Box(0,0,2,2,.8)]


class _FakeBackend:
    def __init__(self,base):
        self.workdir=Path(base)
        self.settings=SimpleNamespace(fps=30.,target_bitrate=3000)
        self.n=2
        self.original=_SyntheticFrame(self.workdir/'input.yuv')
        self.count=0
        self.commands=[]

    def run(self,actions):
        self.count+=1
        self.commands.append(dict(actions))
        stream=self.workdir/f'out{self.count}.266'
        stream.write_bytes(b'fixture_'+str(sorted((p,d) for p,d in actions.items() if d != 0)).encode('ascii'))
        recpath=self.workdir/f'rec{self.count}.yuv';recpath.write_bytes(b'fake_fixture')
        ob0=np.zeros(11,dtype=np.float32)
        ob1=np.zeros(11,dtype=np.float32);ob1[1]=float(actions.get(0,0))
        observations=[({'poc':0},ob0),({'poc':1},ob1)]
        records=[]
        for poc in (0,1):
            records.append({'kind':'picture_action','poc':poc,'qp_native':30,
                            'qp_accepted':30+actions.get(poc,0)})
            records.append({'kind':'coding_qp','poc':poc,'cu_count':2,
                            'qp_min':30+actions.get(poc,0),
                            'qp_max':30+actions.get(poc,0),
                            'qp_mean_unweighted':30+actions.get(poc,0)})
        return SimpleNamespace(observations=observations,records=records,
             bits_by_poc={0:100,1:200},reconstructed=_SyntheticFrame(recpath),
             output_266=stream,encode_seconds=.01,decode_seconds=.01)


class TestStage4InterfaceWithFixtures(unittest.TestCase):
    def test_reset_step_termination_logging(self):
        with tempfile.TemporaryDirectory() as d:
            backend=_FakeBackend(d)
            env=_ReplayEnv(backend,_DeterministicFixtureDetector())
            initial,meta=env.reset()
            self.assertEqual(initial.shape,(11,));self.assertEqual(initial.dtype,np.float32)
            self.assertEqual(meta['mode'],'real_codec_full_sequence_replay_NOT_online')
            obs,reward,done,trunc,info=env.step(3)
            self.assertEqual(backend.commands[-1],{0:1})
            self.assertEqual(info['accepted_delta_qp'],1)
            self.assertEqual(info['frame_index'],0)
            self.assertFalse(done);self.assertFalse(trunc)
            self.assertEqual(obs[1],1)
            self.assertTrue(np.isfinite(reward))
            obs,reward,done,trunc,info=env.step(2)
            self.assertEqual(backend.commands[-1],{0:1,1:0})
            self.assertTrue(done);self.assertFalse(trunc)
            with self.assertRaises(RuntimeError):env.step(2)
            events=(Path(d)/'stage4_rewards.jsonl').read_text().splitlines()
            self.assertEqual(len(events),2)
            self.assertEqual(json.loads(events[0])['frame_index'],0)
            env.close()

    def test_invalid_action_fails_before_codec(self):
        with tempfile.TemporaryDirectory() as d:
            backend=_FakeBackend(d);env=_ReplayEnv(backend,_DeterministicFixtureDetector())
            env.reset()
            with self.assertRaises(ValueError):env.step(8)
            self.assertEqual(len(backend.commands),1)

if __name__=='__main__':unittest.main()
