#!/usr/bin/env python3
"""Reference synchronous dummy-action provider for patched vvencapp.

The program responds to VVRL_OBS lines through vvencapp stdin. It does NOT
compute PPO, decode/reward, nor promise Gymnasium step reward availability.
Use a YUV input FILE, never '--input -' since stdin is the RL control channel.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from state_schema import parse_observation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--encoder',required=True)
    p.add_argument('--action',choices=('zero','plus','minus','alternate','checker-map'),default='zero')
    p.add_argument('--log',required=True)
    p.add_argument('encoder_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    encoder_args=args.encoder_args[1:] if args.encoder_args[:1]==['--'] else args.encoder_args
    if not encoder_args or '-' in encoder_args:
        raise SystemExit('Pass vvencapp arguments after -- and use file input, never stdin input')
    env=os.environ.copy()
    env['VVCRL_MODE']='stdin-map' if args.action=='checker-map' else 'stdin'
    env['VVCRL_LOG']=str(Path(args.log).resolve())
    Path(args.log).unlink(missing_ok=True)
    stderr=Path(args.log+'.encoder_stderr')
    with stderr.open('w',encoding='utf8') as err:
        proc=subprocess.Popen([args.encoder,*encoder_args],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=env)
        n=0
        for line in proc.stdout:
            if not line.startswith('VVCRL_OBS\t'):
                print(line,end='',flush=True)
                continue
            raw,vector=parse_observation(line.split('\t',1)[1])
            poc=raw['poc']
            if args.action=='zero':d=0
            elif args.action=='plus':d=1
            elif args.action=='minus':d=-1
            else:d=1 if (n%2)==0 else -1
            if args.action=='checker-map':
                total=raw['ctu_columns']*raw['ctu_rows']
                csv=','.join(str(1 if i%2==0 else -1) for i in range(total))
                command=f'MAP {poc} {csv}\n'
            else:
                command=f'ACT {poc} {d}\n'
            proc.stdin.write(command); proc.stdin.flush()
            n+=1
            print(f'VVRL POC={poc} state={vector.shape} action={command.strip()[:100]}',flush=True)
        proc.stdin.close()
        rc=proc.wait()
    if rc!=0:
        print('Encoder error log tail:',file=sys.stderr)
        print('\n'.join(stderr.read_text(errors='replace').splitlines()[-30:]),file=sys.stderr)
        raise SystemExit(rc)
    print(f'ENCODE_FINISHED; processed_observations={n}; log={args.log}')

if __name__=='__main__':main()
