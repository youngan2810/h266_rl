#!/usr/bin/env python3
"""Run REAL VVenC + VVdeC + YOLOv5n for a tiny train/val clip.

Requires previously built VVenC with Stage 3 overlay, VVdeC, YOLOv5n source
and weights installed locally. Does not use mock detections or fake bitstreams.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_env.runner import CodecSettings,FullSequenceReplay
from rl_env.detector import LocalYOLOv5n
from rl_env.environment import VVCReplayEnv


def main():
    p=argparse.ArgumentParser()
    for name in ('encoder','decoder','yuv','yolov5-repo','weights','workdir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--fps',type=float,default=30)
    p.add_argument('--bitrate',type=int,default=300000)
    p.add_argument('--decoder-bitdepth',type=int,choices=(8,10),default=8)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--action-index',type=int,choices=range(5),default=2)
    p.add_argument('--iou-threshold',type=float,default=.5)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--nms-iou',type=float,default=.45)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    settings=CodecSettings(encoder=args.encoder,decoder=args.decoder,yuv=args.yuv,
        fps=args.fps,target_bitrate=args.bitrate,decoder_bitdepth=args.decoder_bitdepth)
    detector=LocalYOLOv5n(args.yolov5_repo,args.weights,device=args.device,
                            conf=args.confidence,nms_iou=args.nms_iou)
    env=VVCReplayEnv(FullSequenceReplay(settings,args.workdir),detector,
        iou_threshold=args.iou_threshold)
    try:
        obs,info=env.reset()
        print('RESET',json.dumps({'shape':obs.shape,'dtype':str(obs.dtype),'info':info}))
        if args.steps<=0:raise ValueError('steps must be positive')
        for _ in range(args.steps):
            obs,reward,terminated,truncated,info=env.step(args.action_index)
            print('STEP',json.dumps({'reward':reward,'terminated':terminated,
                                     'truncated':truncated,'info':info},allow_nan=False))
            if terminated:break
    finally:env.close()
    print('RUN_FINISHED; inspect stage4_rewards.jsonl and per-rollout codec logs')

if __name__=='__main__':main()
