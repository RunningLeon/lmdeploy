# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):
    def __init__(self, num_spec_tokens: int):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens

    def get_max_tokens(self, batch_size: int) -> int:
        """Get max tokens."""
        return batch_size * (self.num_spec_tokens + 1)
