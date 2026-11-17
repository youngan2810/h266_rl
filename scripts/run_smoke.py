#!/usr/bin/env python3
"""Run A/B/C/D/E acceptance tests on an ALREADY BUILT stock+patched VVenC pair.

This script is not a proof until it runs in an environment with both actual codec binaries.
It intentionally does NOT run PPO and never modifies x264/partition/MTS code.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np
from generate_yuv420 import generate
from state_schema import parse_observation


def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()


def run_checked(argv, env=None, log=None):
    tick=time.monotonic()
    p=subprocess.run(argv,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if log:Path(log).write_text(p.stdout,encoding='utf8')
    if p.returncode:
        raise RuntimeError(f'Exit={p.returncode}, command={argv}; tail: {p.stdout[-1800:]}')
    return time.monotonic()-tick


def assert_causal_log(records):
    """A referenced feedback must precede its observation in execution log order.
    POC need not increase because VVC may use GOP coding order != display order.
    """
    seen=set()
    for row in records:
        if row.get('kind')=='feedback':seen.add(row['poc'])
        elif row.get('kind')=='observation' and row.get('prev_valid'):
            if row.get('prev_feedback_poc') not in seen:
                raise RuntimeError('State temporal leakage / missing preceding feedback')


def read_jsonl(p):
    if not p.exists():return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def psnr_y(orig,rec,frames,w,h):
    size8=frames*w*h*3//2;size16=size8*2
    if rec.stat().st_size not in (size8,size16):
        raise RuntimeError(f'Decode bytes mismatch: got {rec.stat().st_size}, expected {size8} (8b) or {size16} (16b)')
    a=np.fromfile(orig,dtype=np.uint8)
    yraw=a.reshape(frames,w*h*3//2)[:,:w*h].astype(np.float64)
    if rec.stat().st_size==size8:
        b=np.fromfile(rec,dtype=np.uint8).reshape(frames,w*h*3//2)[:,:w*h].astype(np.float64)
    else:
        b=np.fromfile(rec,dtype='<u2').reshape(frames,w*h*3//2)[:,:w*h].astype(np.float64)/4.0
    mse=np.mean((yraw-b)**2)
    return float('inf') if mse==0 else float(10*np.log10(255*255/mse))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stock',required=True,help='absolute path to stock vvencapp')
    p.add_argument('--patched',required=True,help='absolute path to patched vvencapp')
    p.add_argument('--decoder',help='absolute path to vvdecapp; if omitted, D is NOT verified')
    p.add_argument('--work',default='./vvrl_smoke_results')
    p.add_argument('--frames',type=int,default=12)
    p.add_argument('--bitrate',type=int,default=300000)
    args=p.parse_args()
    work=Path(args.work).resolve();work.mkdir(exist_ok=True,parents=True)
    inp=generate(work/'smoke_480x320_420_8bit.yuv',args.frames)
    settings=['--input',str(inp),'--size','480x320','--fps','30','--frames',str(args.frames),
        '--preset','medium','--bitrate',str(args.bitrate),'--passes','1',
        '--qpa','1','--threads','0','--mtprofile','0','--refreshtype','cra',
        '--intraperiod','32','--additional','GOPSize=32 LookAhead=1 MaxParallelFrames=0']
    results={}
    for label,exe,mode,delta in [('stock',args.stock,None,None),
             ('zero',args.patched,'zero',None),('plus',args.patched,'constant','1'),
             ('minus',args.patched,'constant','-1'),('map',args.patched,'map-checker','1')]:
        out=work/(label+'.266'); log=work/(label+'.jsonl')
        log.unlink(missing_ok=True)
        env=os.environ.copy()
        if mode:
            env.update(VVCRL_MODE=mode,VVCRL_LOG=str(log))
            if delta:env['VVCRL_DELTA']=delta
            else:env.pop('VVCRL_DELTA',None)
        else:
            env.update(VVCRL_MODE='off');env.pop('VVCRL_LOG',None)
        wall=run_checked([exe,*settings,'--output',str(out)],env,work/(label+'.stdout.log'))
        records=read_jsonl(log)
        assert_causal_log(records)
        obs=[x for x in records if x.get('kind')=='observation']
        for ob in obs:parse_observation(ob)
        actions=[x for x in records if x.get('kind')=='picture_action']
        ctu_actions=[x for x in records if x.get('kind')=='ctu_map_action']
        coded_ctu=[x for x in records if x.get('kind')=='coding_ctu_qp']
        if mode and not obs:raise RuntimeError(f'{label}: no observations logged')
        if mode and len(actions)!=len(obs):raise RuntimeError(f'{label}: action count != observations')
        if label=='zero' and any(x['qp_native']!=x['qp_accepted'] for x in actions):
            raise RuntimeError('No-op action unexpectedly changed picture QP')
        if label=='plus' and not any(x['qp_accepted']>x['qp_native'] for x in actions):
            raise RuntimeError('Positive action failed to change picture QP')
        if label=='minus' and not any(x['qp_accepted']<x['qp_native'] for x in actions):
            raise RuntimeError('Negative action failed to change picture QP')
        if label=='map' and not ctu_actions:raise RuntimeError('No CTU map actions logged')
        if label=='map' and not coded_ctu:raise RuntimeError('No post-encode CTU coding QP measurements')
        duration=args.frames/30
        results[label]={"sha256":filehash(out),"bytes":out.stat().st_size,
            "actual_kbps":out.stat().st_size*8/duration/1000,
            "bpp":out.stat().st_size*8/(args.frames*480*320),
            "wall_seconds":wall,"n_observations":len(obs),
            "n_ctu_actions":len(ctu_actions),"n_coded_ctu":len(coded_ctu)}
        if args.decoder:
            decoded=work/(label+'_decoded.yuv')
            results[label]['decoder_wall_seconds']=run_checked(
                [args.decoder,'-b',str(out),'-o',str(decoded)],log=work/(label+'.decode.log'))
            results[label]['decoded_bytes']=decoded.stat().st_size
            results[label]['psnr_y']=psnr_y(inp,decoded,args.frames,480,320)
        print(label,results[label],flush=True)
    results['test_A_noop_bit_identical']=results['stock']['sha256']==results['zero']['sha256']
    if not results['test_A_noop_bit_identical']:
        raise RuntimeError('TEST A FAILED: ΔQP=0 bitstream differs from stock; inspect logs/config')
    results['test_D_decode_verified']=bool(args.decoder)
    Path(work/'report.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print('SMOKE RESULT:',work/'report.json')
    if not args.decoder:print('DECODE NOT VERIFIED. Supply --decoder to complete Test D.')

if __name__=='__main__':main()
