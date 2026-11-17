"""Stage 4 VVC-native task-aware RL research modules (pre-PPO)."""
from .environment import VVCReplayEnv
from .reward import Box, InvalidReward, compose_reward, precision_proxy
from .runner import CodecFailure, CodecSettings, FullSequenceReplay
__all__=['VVCReplayEnv','Box','InvalidReward','compose_reward','precision_proxy',
         'CodecFailure','CodecSettings','FullSequenceReplay']
