#!/usr/bin/env python3
"""Convert one permitted BDD100K training/validation video to strict 480x320 YUV420p.

Input/output and ffmpeg version are recorded locally. Never commit original
videos, converted YUV or result logs to GitHub.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--frames',type=int,default=8)
    args=p.parse_args()
    inp=Path(args.input).resolve();out=Path(args.output).resolve()
    if not inp.is_file() or args.fps<=0 or args.frames<=0:raise ValueError('invalid input/fps/frames')
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(inp),
         '-vf',f'fps={args.fps},scale=480:320:flags=bicubic',
         '-frames:v',str(args.frames),'-an','-pix_fmt','yuv420p','-f','rawvideo',str(out)]
    subprocess.run(cmd,check=True,timeout=300)
    frame_bytes=480*320*3//2
    if out.stat().st_size!=args.frames*frame_bytes:
        raise RuntimeError('conversion did not produce the expected frame count')
    meta={'source_video':str(inp),'output_yuv':str(out),'fps':args.fps,
          'frames':args.frames,'dimensions':[480,320],'pix_fmt':'yuv420p','command':cmd}
    (out.parent/(out.stem+'.prepare.json')).write_text(json.dumps(meta,indent=2))
    print('PREPARED',out,'frames',args.frames)

if __name__=='__main__':main()
