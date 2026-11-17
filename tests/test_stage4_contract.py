"""Interface/format tests; no fake passing VVenC or YOLO sessions."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from rl_env.video import YUV420,VideoFormatError
from rl_env.runner import CodecSettings,CodecFailure
from rl_env.environment import _ReplayEnv
from scripts.state_schema import action_from_policy_index

class TestStage4Contract(unittest.TestCase):
    def test_action_mapping_preserved(self):
        self.assertEqual([action_from_policy_index(x) for x in range(5)],[-2,-1,0,1,2])
    def test_missing_binaries_fail(self):
        with self.assertRaises(CodecFailure):CodecSettings('/missing/encoder','/missing/decoder','/missing/yuv',30,300000).verify()
    def test_yuv_strict_size_and_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'video.yuv';p.write_bytes(bytes(4*2*3//2))
            stream=YUV420(p,4,2,8)
            self.assertEqual(stream.frame_count,1)
            x=stream.rgb(0)
            self.assertEqual(x.shape,(2,4,3));self.assertEqual(x.dtype,np.uint8)
            with self.assertRaises(VideoFormatError):stream.rgb(1)
    def test_yuv_rejects_partial_frame(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.yuv';p.write_bytes(b'abc')
            with self.assertRaises(VideoFormatError):YUV420(p,4,2,8)
    def test_no_uninitialized_step(self):
        # Skip constructing external codec to test API's fail-closed transition.
        obj=object.__new__(_ReplayEnv)
        obj.current=None;obj.terminated=False
        with self.assertRaises(RuntimeError):obj.step(2)

if __name__=='__main__':unittest.main()
