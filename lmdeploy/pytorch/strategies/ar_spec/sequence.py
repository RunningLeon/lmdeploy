# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from torch import Tensor

from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import InputEmbeddings, MessageStatus, MultiModalInputs, UpdateTokenMode, _to_ndarray

from ..ar.sequence import ARSequenceStrategy, SchedulerSequenceDefault

SeqList = List['SchedulerSequenceARSpec']


@dataclass
class SchedulerSequenceARSpec(SchedulerSequenceDefault):
    # spec decode
    draft_token_ids: np.ndarray = np.empty(0, dtype=np.int64)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         draft_token_ids: Tensor = None,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        if draft_token_ids is not None:
            self.draft_token_ids = _to_ndarray(draft_token_ids)
        super().update_token_ids(token_ids,
                                 multimodals=multimodals,
                                 embeddings=embeddings,
                                 model_meta=model_meta,
                                 mode=mode,
                                 **kwargs)


class ARSpecSequenceStrategy(ARSequenceStrategy):

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, is_decoding: bool) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        draft_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)
        stop_pos = batched_outputs.stop_pos

        batch_size = len(running)
        next_token_ids = next_token_ids.view(batch_size, -1).numpy()
        draft_token_ids = draft_token_ids.view(batch_size, -1).numpy()
        stop_pos = stop_pos.tolist()
        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for idx, token in enumerate(next_token_ids):
            msg = running[idx]
            stop = stopped[idx]
            model_meta = model_metas[idx]
            if msg.status != MessageStatus.LOCKED:
                continue
            token = token[token > -1]

            cur_draft_tokens = draft_token_ids[idx] if not stop else np.empty(0, dtype=np.int64)
            # fill token
            msg.update_token_ids(token, draft_token_ids=cur_draft_tokens, model_meta=model_meta, mode=update_mode)
            if stop:
                msg.set_stop_pos(stop_pos[idx])
                msg.status = MessageStatus.TO_BE_MIGRATED if msg.preserve_cache else MessageStatus.STOPPED
