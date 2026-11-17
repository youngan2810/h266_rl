"""YOLOv5n local-only adapter: no network, no weight auto-download."""
from pathlib import Path
import hashlib
from .reward import Box


class DetectorError(RuntimeError):pass


class LocalYOLOv5n:
    def __init__(self, yolov5_repo, weights, device='cpu', conf=.25, nms_iou=.45, size=640):
        source=Path(yolov5_repo).resolve(); weights=Path(weights).resolve()
        if not (source/'hubconf.py').exists():raise DetectorError('local YOLOv5 repo/hubconf.py missing')
        if not weights.is_file():raise DetectorError('YOLOv5n weights missing; download separately, do not commit')
        if not (0<conf<=1 and 0<nms_iou<=1 and size>0):raise DetectorError('invalid YOLO settings')
        try:import torch
        except ImportError as exc:raise DetectorError('PyTorch required') from exc
        self.model=torch.hub.load(str(source),'custom',path=str(weights),source='local',trust_repo=True)
        self.model=self.model.to(device).eval()
        self.model.conf=float(conf);self.model.iou=float(nms_iou);self.model.classes=[2]
        names=self.model.names
        car=names[2] if isinstance(names,(list,tuple)) else names.get(2)
        if car!='car':raise DetectorError('weights are not COCO car-class compatible')
        self.size=int(size)
        self.sha256=hashlib.sha256(weights.read_bytes()).hexdigest()
        self.config={'weights_sha256':self.sha256,'conf':conf,'nms_iou':nms_iou,'size':size,'device':device}

    def detect(self,rgb):
        import numpy as np
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise DetectorError('expected HxWx3 uint8 RGB')
        try:
            results=self.model(rgb,size=self.size)
            arr=results.xyxy[0].detach().cpu().numpy()
        except Exception as exc:raise DetectorError('YOLO inference failed') from exc
        return [Box(float(x1),float(y1),float(x2),float(y2),float(conf),int(cls))
                for x1,y1,x2,y2,conf,cls in arr if int(cls)==2]
