#!/usr/bin/env python3
"""Generate deterministic small YUV420p 8-bit smoke video (moving rectangle)."""
from pathlib import Path
import argparse
import numpy as np

def generate(output, frames=12, width=480, height=320):
    if width%2 or height%2 or frames<=0: raise ValueError('YUV420 needs even dims and >0 frames')
    y, x = np.mgrid[0:height,0:width]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as f:
        for i in range(frames):
            frame=np.asarray((x//4+y//6+i*2)%170+25,dtype=np.uint8)
            px=(i*8)%(width-80)
            frame[height//3:height//3+52,px:px+80]=230
            u=np.full((height//2,width//2),120,dtype=np.uint8)
            v=np.full((height//2,width//2),136,dtype=np.uint8)
            f.write(frame.tobytes());f.write(u.tobytes());f.write(v.tobytes())
    assert output.stat().st_size==frames*width*height*3//2
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--width',type=int,default=480);p.add_argument('--height',type=int,default=320)
    args=p.parse_args();print(generate(**vars(args)))
