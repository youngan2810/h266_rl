"""Paper-grounded rate reward and explicit, configurable car-precision proxy.

CVPR 2025 RL-RC-DoT: r_rate=-abs(log(current_average_rate/target_rate)),
r_total=r_rate+lambda*r_task; appendix reports lambda=20. Paper does not
fully specify association/confidence/IoU/empty-detection semantics. The
matching policy here is an explicit reproducibility choice, NOT claimed
bit-for-bit replication of its detector evaluation.
"""
from dataclasses import dataclass
import math
from typing import Sequence


class InvalidReward(ValueError):
    """Inputs do not permit a scientifically valid reward."""


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: int = 2  # COCO car

    def __post_init__(self):
        a = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        if not all(math.isfinite(v) for v in a):
            raise InvalidReward('nonfinite detection')
        if self.x2 <= self.x1 or self.y2 <= self.y1 or not 0 <= self.confidence <= 1:
            raise InvalidReward('invalid bounding box or confidence')
        if self.label != 2:
            raise InvalidReward('only COCO car class 2 is allowed')


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    area = max(0., x2-x1)*max(0., y2-y1)
    union = (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-area
    return area/union if union > 0 else 0.


def precision_proxy(raw: Sequence[Box], reconstructed: Sequence[Box], iou_threshold: float = .5) -> tuple[float, int, int]:
    """One-to-one greedy confidence-ordered matching, derived explicit convention.

    Fails closed for empty raw pseudo-labels or empty reconstructed predictions;
    the latter has undefined mathematical precision, and is *not* set to zero
    silently. Choose and document a separate missing-detection policy before PPO.
    """
    if not (math.isfinite(iou_threshold) and 0 < iou_threshold <= 1):
        raise InvalidReward('IoU threshold must be in (0,1]')
    if not raw or not reconstructed:
        raise InvalidReward('undefined precision for empty raw/reconstructed car detections')
    used: set[int] = set()
    hits = 0
    for det in sorted(reconstructed, key=lambda x: x.confidence, reverse=True):
        choices = [(iou(det, gt), j) for j, gt in enumerate(raw) if j not in used]
        if not choices:
            continue
        score, index = max(choices)
        if score >= iou_threshold:
            used.add(index)
            hits += 1
    false_positives = len(reconstructed)-hits
    return hits/len(reconstructed), hits, false_positives


def current_average_bitrate(actual_bits_by_poc: dict[int,int], frame_index: int, fps: float) -> float:
    """Prefix of presentation POCs [0..frame_index], never includes later POCs."""
    if frame_index < 0 or not math.isfinite(fps) or fps <= 0:
        raise InvalidReward('invalid frame index or fps')
    expected = set(range(frame_index+1))
    if not expected.issubset(actual_bits_by_poc):
        raise InvalidReward('missing completed frame bit counts in bitrate prefix')
    bits = [actual_bits_by_poc[i] for i in range(frame_index+1)]
    if any(type(b) is not int or b < 0 for b in bits):
        raise InvalidReward('invalid actual bits')
    rate = fps*sum(bits)/(frame_index+1)
    if not math.isfinite(rate) or rate <= 0:
        raise InvalidReward('zero/invalid current average bitrate')
    return rate


def compose_reward(actual_bitrate: float, target_bitrate: float, task_precision: float, lam: float = 20.):
    if not all(math.isfinite(x) for x in (actual_bitrate, target_bitrate, task_precision, lam)):
        raise InvalidReward('nonfinite reward argument')
    if actual_bitrate <= 0 or target_bitrate <= 0 or not 0 <= task_precision <= 1 or lam < 0:
        raise InvalidReward('invalid reward argument range')
    r_rate = -abs(math.log(actual_bitrate/target_bitrate))
    r_task = task_precision
    return {'rate_reward':r_rate, 'task_reward':r_task, 'total_reward':r_rate+lam*r_task}
