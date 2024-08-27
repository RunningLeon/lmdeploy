# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

apply_rotary_pos_emb = FunctionDispatcher('apply_rotary_pos_emb').make_caller()
apply_rotary_pos_emb_longcache = FunctionDispatcher(
    'apply_rotary_pos_emb_longcache').make_caller()
