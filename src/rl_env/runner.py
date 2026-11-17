"""Real (not simulated) VVenC/VVdeC full-sequence replay backend.

Each rollout encodes the *entire* contiguous raw sequence in ONE encoder process,
using past chosen actions and zero for undecided future actions. Replay is O(T^2),
NOT true online one-picture step and NOT suitable for scalable PPO yet.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import subprocess
import threading
import time

from scripts.state_schema import parse_observation
from .video import YUV420, VideoFormatError


class CodecFailure(RuntimeError):pass


@dataclass(frozen=True)
class CodecSettings:
    encoder: str
    decoder: str
    yuv: str
    fps: float
    target_bitrate: int
    width: int=480
    height: int=320
    max_frames: int=0
    preset: str='medium'
    decoder_bitdepth: int=8
    timeout_seconds: float=1800
    # Fixed Stage 3 serial rate-control/QPA pilot. No hidden bypass.

    def verify(self):
        for k in ('encoder','decoder','yuv'):
            if not Path(getattr(self,k)).is_file():raise CodecFailure(f'{k} not found: {getattr(self,k)}')
        if self.fps<=0 or self.target_bitrate<=0 or self.width<=0 or self.height<=0:
            raise CodecFailure('invalid video/bitrate settings')
        if self.max_frames<0 or self.decoder_bitdepth not in (8,10) or self.timeout_seconds<=0:
            raise CodecFailure('invalid frame count, decoder bitdepth or timeout')
        if self.preset not in {'faster','fast','medium','slow','slower','medium_lowDecEnergy'}:
            raise CodecFailure('unrecognized VVenC preset')


@dataclass
class Rollout:
    observations: list[tuple[dict,object]]
    records: list[dict]
    bits_by_poc: dict[int,int]
    reconstructed: YUV420
    encode_seconds: float
    decode_seconds: float
    output_266: Path


def _read_jsonl_strict(p: Path):
    if not p.is_file():raise CodecFailure('encoder did not produce a control log')
    rows=[]
    for n,line in enumerate(p.read_text(encoding='utf8').splitlines(),1):
        try:r=json.loads(line)
        except Exception as exc:raise CodecFailure(f'invalid JSONL line {n}') from exc
        if not isinstance(r,dict):raise CodecFailure(f'non-object JSONL line {n}')
        rows.append(r)
    return rows


class FullSequenceReplay:
    """Verification-only backend; no claim of causal online Gym environment."""
    def __init__(self, settings: CodecSettings, workdir):
        settings.verify()
        self.settings=settings
        self.original=YUV420(settings.yuv,settings.width,settings.height,8)
        n=self.original.frame_count
        if settings.max_frames and settings.max_frames!=n:
            raise CodecFailure(f'input has {n} frames but configured max_frames={settings.max_frames}; pass exactly one prepared sequence')
        self.n=n
        self.workdir=Path(workdir).resolve()
        self.workdir.mkdir(parents=True,exist_ok=True)
        self.rollout_number=0

    def _encode(self, actions:dict[int,int], bitstream:Path, logfile:Path):
        s=self.settings
        argv=[s.encoder,'--input',str(Path(s.yuv).resolve()),'--size',f'{s.width}x{s.height}',
              '--fps',f'{s.fps:g}','--frames',str(self.n),'--preset',s.preset,
              '--bitrate',str(s.target_bitrate),'--passes','1','--qpa','1',
              '--threads','0','--mtprofile','0','--refreshtype','cra',
              '--intraperiod','32','--additional','GOPSize=32 LookAhead=1 MaxParallelFrames=0',
              '--output',str(bitstream)]
        env=os.environ.copy();env.update(VVCRL_MODE='stdin',VVCRL_LOG=str(logfile))
        env.pop('VVCRL_DELTA',None);env.pop('VVCRL_MAP',None)
        stderr_path=bitstream.with_suffix('.encoder.stderr.txt')
        observations=[];seen_poc=set();event_error=None
        t0=time.perf_counter()
        with stderr_path.open('w',encoding='utf8') as err:
            proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=err,text=True,bufsize=1,env=env)
            expired=threading.Event()
            def _terminate():expired.set();proc.kill()
            timer=threading.Timer(s.timeout_seconds,_terminate);timer.daemon=True;timer.start()
            try:
                for line in proc.stdout:
                    if not line.startswith('VVCRL_OBS\t'):
                        continue
                    data,vec=parse_observation(line.split('\t',1)[1])
                    poc=data['poc']
                    if poc in seen_poc or not 0<=poc<self.n:
                        raise CodecFailure(f'duplicate/out-of-range POC: {poc}')
                    seen_poc.add(poc)
                    delta=actions.get(poc,0)
                    if type(delta) is not int or delta not in (-2,-1,0,1,2):
                        raise CodecFailure('invalid action delta in replay')
                    proc.stdin.write(f'ACT {poc} {delta}\n');proc.stdin.flush()
                    observations.append((data,vec))
                ret=proc.wait()
            except BaseException as exc:
                event_error=exc;proc.kill();proc.wait();ret=proc.returncode
            finally:
                timer.cancel()
                if proc.stdin and not proc.stdin.closed:proc.stdin.close()
                if proc.stdout and not proc.stdout.closed:proc.stdout.close()
        if event_error:raise CodecFailure(f'encoder observation protocol failed: {event_error}') from event_error
        if expired.is_set():raise CodecFailure('VVenC timed out')
        if ret!=0:
            tail='\n'.join(stderr_path.read_text(errors='replace').splitlines()[-15:])
            raise CodecFailure(f'VVenC exited {ret}: {tail}')
        if len(observations)!=self.n or set(x['poc'] for x,_ in observations)!=set(range(self.n)):
            raise CodecFailure(f'observation count mismatch: {len(observations)} vs {self.n}')
        if not bitstream.is_file() or bitstream.stat().st_size==0:raise CodecFailure('empty/missing .266 output')
        return observations,time.perf_counter()-t0

    def run(self,actions:dict[int,int]):
        unknown=set(actions)-set(range(self.n))
        if unknown:raise CodecFailure(f'action POCs outside video: {sorted(unknown)}')
        self.rollout_number+=1
        directory=self.workdir/f'rollout_{self.rollout_number:04d}';directory.mkdir(exist_ok=False)
        bitstream=directory/'out.266';logfile=directory/'control.jsonl'
        observations,encode_seconds=self._encode(actions,bitstream,logfile)
        rows=_read_jsonl_strict(logfile)
        acts={}; feedback={}; slices={}; coded_qp={}; observed=set();seen_feedback=set()
        for row in rows:
            kind=row.get('kind');poc=row.get('poc')
            if kind=='observation':
                observed.add(poc)
                if row.get('prev_valid') and row.get('prev_feedback_poc') not in seen_feedback:
                    raise CodecFailure(f'potential future feedback leakage in POC={poc}')
            elif kind=='picture_action':
                if poc in acts:raise CodecFailure('duplicate picture_action event')
                acts[poc]=row
            elif kind=='slice_qp':
                slices[poc]=row
            elif kind=='coding_qp':
                if poc in coded_qp:raise CodecFailure('duplicate coding_qp summary')
                if type(row.get('cu_count')) is not int or row['cu_count']<=0:
                    raise CodecFailure('invalid CU coding count')
                try:
                    finite=math.isfinite(float(row['qp_mean_unweighted']))
                    valid_bounds=row['qp_min']<=row['qp_max']
                except (KeyError,TypeError,ValueError):
                    raise CodecFailure('malformed coding_qp summary')
                if not finite or not valid_bounds:raise CodecFailure('invalid coded QP statistics')
                coded_qp[poc]=row
            elif kind=='feedback':
                if poc in feedback:raise CodecFailure('duplicate completed picture feedback')
                if type(row.get('actual_bits')) is not int or row['actual_bits']<0:
                    raise CodecFailure('invalid bit count')
                feedback[poc]=row['actual_bits'];seen_feedback.add(poc)
        if set(acts)!=set(range(self.n)) or set(feedback)!=set(range(self.n)):
            raise CodecFailure('missing per-POC picture action / actual bits; cannot compute reward')
        if not set(slices).issuperset(acts):raise CodecFailure('missing post-QPA slice QP trace')
        if set(coded_qp)!=set(range(self.n)):
            raise CodecFailure('missing actual coded-CU QP summary: cannot verify action effect')
        for poc,row in acts.items():
            ob=next(item for item,_ in observations if item['poc']==poc)
            d=actions.get(poc,0)
            expected=ob['native_qp']+d
            clipped=max(ob['qp_min'],min(ob['qp_max'],expected))
            if row.get('qp_requested')!=expected or row.get('qp_accepted')!=clipped:
                raise CodecFailure(f'action not correctly applied for POC {poc}')
        decoded_file=directory/'decoded.yuv'
        t0=time.perf_counter()
        with (directory/'decode.stdout.txt').open('w',encoding='utf8') as out:
            try:subprocess.run([self.settings.decoder,'-b',str(bitstream),'-o',str(decoded_file)],
                stdout=out,stderr=subprocess.STDOUT,check=True,timeout=self.settings.timeout_seconds)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                raise CodecFailure('VVdeC decode failed; inspect decode.stdout.txt') from exc
        decode_seconds=time.perf_counter()-t0
        try: rec=YUV420(decoded_file,self.settings.width,self.settings.height,self.settings.decoder_bitdepth)
        except VideoFormatError as exc:raise CodecFailure('decoded YUV format mismatch') from exc
        if rec.frame_count!=self.n:raise CodecFailure(f'decoded count {rec.frame_count} != source {self.n}')
        return Rollout(observations,rows,feedback,rec,encode_seconds,decode_seconds,bitstream)
