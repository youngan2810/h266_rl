"""Strict raw 8/10-bit planar YUV420 frame reader, with explicit POC indexing."""
from pathlib import Path
import numpy as np


class VideoFormatError(ValueError):
    pass


class YUV420:
    def __init__(self, path, width=480, height=320, bitdepth=8):
        self.path=Path(path)
        self.width=int(width);self.height=int(height);self.bitdepth=int(bitdepth)
        if self.width<=0 or self.height<=0 or self.width%2 or self.height%2 or self.bitdepth not in (8,10):
            raise VideoFormatError('expected even positive W/H and 8/10-bit YUV420')
        if not self.path.is_file():raise VideoFormatError(f'file not found: {self.path}')
        self.frame_bytes=self.width*self.height*3//2*(1 if self.bitdepth==8 else 2)
        size=self.path.stat().st_size
        if size==0 or size%self.frame_bytes:
            raise VideoFormatError(f'file size {size} not divisible by frame bytes {self.frame_bytes}')
        self.frame_count=size//self.frame_bytes

    def rgb(self, poc: int):
        """Decode one frame to RGB uint8, unchanged input dimensions."""
        if type(poc) is not int or poc<0 or poc>=self.frame_count:raise VideoFormatError('POC outside frame count')
        try:import cv2
        except ImportError as exc:raise VideoFormatError('opencv-python required') from exc
        with self.path.open('rb') as f:
            f.seek(poc*self.frame_bytes)
            data=f.read(self.frame_bytes)
        if len(data)!=self.frame_bytes:raise VideoFormatError('short YUV frame')
        if self.bitdepth==8:
            arr=np.frombuffer(data,dtype=np.uint8)
        else:
            words=np.frombuffer(data,dtype='<u2')
            if np.any(words>1023):raise VideoFormatError('non-10-bit range in 16-bit words')
            arr=(words>>2).astype(np.uint8)
        planar=arr.reshape((self.height*3//2,self.width))
        return cv2.cvtColor(planar,cv2.COLOR_YUV2RGB_I420)
