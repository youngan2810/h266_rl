import json
from pathlib import Path
import subprocess
import sys
import os
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from state_schema import parse_observation, action_from_policy_index

ROOT=Path(__file__).resolve().parents[1]

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.item=dict(schema='vvrl.v1',kind='observation',poc=0,native_qp=27,qp_min=0,qp_max=63,
                frame_type=2,temporal_layer=3,gop_position=0,gop_size=32,target_bpp=.066,
                budget_error=0.,prev_qp=0,prev_bit_ratio=0.,prev_valid=False,prev_feedback_poc=-1,picture_target_bits=500,
                ctu_columns=4,ctu_rows=3,source_width=480,source_height=320,
                spatial_activity=0.,temporal_activity=0.,spatial_valid=False,temporal_valid=False)
    def test_shape(self):
        o,v=parse_observation(self.item)
        self.assertEqual(v.shape,(11,))
        self.assertEqual(v.dtype.name,'float32')
        self.assertTrue((v==v).all())
    def test_bad_nan(self):
        self.item['budget_error']=float('nan')
        with self.assertRaises(ValueError):parse_observation(self.item)
    def test_bad_qp(self):
        self.item['native_qp']=64
        with self.assertRaises(ValueError):parse_observation(self.item)
    def test_prev_missing(self):
        self.item['prev_bit_ratio']=1.
        with self.assertRaises(ValueError):parse_observation(self.item)
    def test_action_range(self):
        self.assertEqual(action_from_policy_index(0),-2)
        self.assertEqual(action_from_policy_index(4),2)
        with self.assertRaises(ValueError):action_from_policy_index(5)
    def test_live_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'test_bridge'
            subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'tests'/'test_control.cpp'),'-o',str(exe)],check=True)
            e=os.environ.copy();e['VVCRL_MODE']='stdin';e['VVCRL_LOG']=str(Path(td)/'state.jsonl')
            p=subprocess.run([str(exe)],input='ACT 7 -1\n',env=e,text=True,capture_output=True,check=True)
            self.assertTrue(p.stdout.startswith('VVCRL_OBS\t'))
            self.assertIn('TEST_DELTA=-1',p.stdout)
            obs=json.loads(p.stdout.splitlines()[0].split('\t',1)[1])
            self.assertEqual(parse_observation(obs)[1].shape,(11,))

    def test_live_protocol_rejects_wrong_poc(self):
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'test_bridge'
            subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'tests'/'test_control.cpp'),'-o',str(exe)],check=True)
            e=os.environ.copy();e['VVCRL_MODE']='stdin'
            p=subprocess.run([str(exe)],input='ACT 8 -1\n',env=e,text=True,capture_output=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('VVRL expected ACT',p.stderr)

    def test_live_map_shape(self):
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'test_bridge'
            subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'tests'/'test_control.cpp'),'-o',str(exe)],check=True)
            e=os.environ.copy();e['VVCRL_MODE']='stdin-map'
            action='MAP 7 '+','.join('1' if i%2==0 else '-1' for i in range(12))+'\n'
            p=subprocess.run([str(exe)],input=action,env=e,text=True,capture_output=True,check=True)
            self.assertIn('TEST_MAP_0=1 TEST_MAP_1=-1',p.stdout)
            bad=subprocess.run([str(exe)],input='MAP 7 1,1\n',env=e,text=True,capture_output=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('map shape mismatch',bad.stderr)

    def test_map_file_rejects_invalid_ctu(self):
        with tempfile.TemporaryDirectory() as td:
            exe=Path(td)/'test_bridge'
            subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'tests'/'test_control.cpp'),'-o',str(exe)],check=True)
            m=Path(td)/'map.csv';m.write_text('7,12,1\n')
            e=os.environ.copy();e.update(VVCRL_MODE='map-file',VVCRL_MAP=str(m))
            p=subprocess.run([str(exe)],env=e,text=True,capture_output=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('outside grid',p.stderr)

    def test_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=ROOT/'tests'/'_control_smoke'
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',
                str(ROOT/'tests'/'test_control.cpp'),'-o',str(exe)],check=True)
            try:
                for mode,delta in [('zero',0),('constant',2),('map-checker',0)]:
                    log=Path(tmp)/(mode+'.jsonl');env=os.environ.copy()
                    env.update(VVCRL_MODE=mode,VVCRL_LOG=str(log),VVCRL_DELTA='2')
                    p=subprocess.run([str(exe)],env=env,text=True,capture_output=True,check=True)
                    self.assertIn('TEST_DELTA='+str(delta),p.stdout)
                    records=[json.loads(l) for l in log.read_text().splitlines()]
                    self.assertEqual(records[0]['schema'],'vvrl.v1')
                    self.assertEqual(records[-1]['kind'],'feedback')
                    if mode=='map-checker':self.assertIn('TEST_MAP_0=2 TEST_MAP_1=-2',p.stdout)
            finally:exe.unlink(missing_ok=True)

if __name__=='__main__':unittest.main()
