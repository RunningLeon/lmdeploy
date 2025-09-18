# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens

    def get_max_tokens(self, batch_size: int, input_ids: torch.Tensor) -> int:
        """Get max tokens."""
        num_tokens = input_ids.size(1)
        assert num_tokens % batch_size == 0, 'The input_ids length must be divisible by batch_size.'
        return num_tokens
