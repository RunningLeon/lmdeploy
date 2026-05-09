# Copyright (c) OpenMMLab. All rights reserved.
import inspect

from lmdeploy.pytorch.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor


def test_qwen3_5_mtp_fusion_projection_uses_active_dist_context():
    source = inspect.getsource(Qwen3_5MultiTokenPredictor.__init__)

    assert 'is_tp=' not in source
    assert 'dp_disable_tp=' not in source
