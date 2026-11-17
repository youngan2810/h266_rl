"""Deterministic mathematics-only checks. NOT evidence of codec/detector E2E."""
import math
import unittest
from rl_env.reward import Box,InvalidReward,compose_reward,current_average_bitrate,precision_proxy,iou


def b(a=0, conf=.9):return Box(a,a,a+10,a+10,conf)

class TestStage4Reward(unittest.TestCase):
    def test_rate_equal_target(self):
        r=compose_reward(100000,100000,1.0)
        self.assertEqual(r['rate_reward'],0.0)
        self.assertEqual(r['total_reward'],20.0)
    def test_log_symmetry(self):
        self.assertAlmostEqual(compose_reward(200,100,0)['rate_reward'],-math.log(2))
        self.assertAlmostEqual(compose_reward(50,100,0)['rate_reward'],-math.log(2))
    def test_prefix_bitrate_no_future_bits(self):
        bits={0:100,1:300,2:999999}
        self.assertEqual(current_average_bitrate(bits,1,30),6000)
    def test_missing_poc_fails(self):
        with self.assertRaises(InvalidReward):current_average_bitrate({0:100,2:200},1,30)
    def test_zero_bitrate_fails(self):
        with self.assertRaises(InvalidReward):compose_reward(0,100,0.5)
        with self.assertRaises(InvalidReward):current_average_bitrate({0:0},0,30)
    def test_precision_tp_fp(self):
        p,tp,fp=precision_proxy([b(0)],[b(0),b(30,.7)])
        self.assertEqual((p,tp,fp),(.5,1,1))
    def test_one_to_one(self):
        p,tp,fp=precision_proxy([b(0)],[b(0),b(0,.8)])
        self.assertEqual((p,tp,fp),(.5,1,1))
    def test_iou_threshold(self):
        self.assertEqual(iou(b(0),b(0)),1)
        p,tp,fp=precision_proxy([b(0)],[b(30)])
        self.assertEqual((p,tp,fp),(0,0,1))
    def test_empty_predictions_explicit_invalid(self):
        for raw,rec in [([],[]),([b()],[]),([],[b()])]:
            with self.assertRaises(InvalidReward):precision_proxy(raw,rec)
    def test_bad_boxes_and_parameters(self):
        with self.assertRaises(InvalidReward):Box(1,1,1,2,.7)
        with self.assertRaises(InvalidReward):Box(0,0,1,1,float('nan'))
        with self.assertRaises(InvalidReward):compose_reward(10,10,1.1)
        with self.assertRaises(InvalidReward):precision_proxy([b()],[b()],0)

if __name__=='__main__':unittest.main()
