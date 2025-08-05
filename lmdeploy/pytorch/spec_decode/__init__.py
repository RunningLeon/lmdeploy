# Copyright (c) OpenMMLab. All rights reserved.

from .base import build_specdecode_proposer
from .deepseek_mtp import DeepseekMTP  # noqa F401
from .eagle import Eagle  # noqa F401
from .eagle3 import Eagle3  # noqa F401
from .reject_sampler import RejectionSampler

__all__ = ['RejectionSampler', 'build_specdecode_proposer']
