# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import (InputEmbeddings, MessageStatus, MultiModalInputs, SamplingParam,
                                       SchedulerSequence, SchedulerSession, UpdateTokenMode, _to_ndarray)

from ..base.sequence import SequenceStrategy

SeqList = List[SchedulerSequence]


@dataclass
class SchedulerSequenceDefault(SchedulerSequence):

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         mode: UpdateTokenMode = UpdateTokenMode.INPUTS,
                         expert_ids: Tensor = None,
                         **kwargs):
        """Update token ids, old token ids will be added to history."""
        # update history image nums
        self._update_embeddings(embeddings)

        # update multimodals
        self._update_multimodals(multimodals)

        token_ids = _to_ndarray(token_ids)

        num_valid = len(token_ids)
        # record cached expert ids
        if self.output_expert_ids:
            if expert_ids is not None:
                if self.all_experts_ids is None:
                    self.all_experts_ids = expert_ids
                else:
                    self.all_experts_ids = torch.cat([self.all_experts_ids, expert_ids], dim=0)

        if mode == UpdateTokenMode.INPUTS:
            self.arrive_time = time.perf_counter()
            self.output_start_pos = self.num_all_ids + len(token_ids)
            self._num_token_ids += num_valid
            self.num_new_tokens = 0
        else:
            self._num_history_ids += self._num_token_ids
            num_token_ids = num_valid
            self._num_token_ids = num_token_ids
            self.num_new_tokens += num_token_ids

        self.history_cache.append(token_ids)

        if model_meta is not None:
            self.model_meta = model_meta

    def set_step(self, step: int):
        """Set step."""
        num_all_ids = self.num_all_ids
        # update step for vlm
        if len(self.history_embeddings) > 0:
            new_step, self._num_history_images, self._num_images = \
                self.history_embeddings.get_step(step)
            assert 0 <= new_step <= step
            step = new_step
        self._num_history_ids = step
        self._num_token_ids = num_all_ids - step
        self.num_ignored_history = min(step, self.num_ignored_history)

        self.model_meta = None

        # cross
        if self.history_multimodals is not None:
            self._num_history_cross = self.history_multimodals.get_encoder_len(0, self.num_history_ids)
            self._num_cross = self.history_multimodals.get_encoder_len(self._num_history_ids, num_all_ids)

        if self.output_expert_ids:
            self.all_experts_ids = None


class ARSequenceStrategy(SequenceStrategy):

    def __init__(self, model_config: Any):
        """config."""
        self.model_config = model_config

    def make_sequence(self,
                      seq_id: int,
                      session: 'SchedulerSession',
                      sampling_param: 'SamplingParam' = None,
                      adapter_name: str = None,
                      migration_request: Optional[MigrationRequest] = None,
                      resp_cache: bool = False,
                      preserve_cache: bool = False) -> 'SchedulerSequence':
        """Make sequence."""
        num_moe_layers = None
        num_experts_per_tok = None
        if getattr(self.model_config.hf_config, 'cache_expert_ids', False):
            num_experts_per_tok = self.model_config.hf_config.num_experts_per_tok
            num_moe_layers = self.model_config.num_layers
        return SchedulerSequenceDefault(
            seq_id=seq_id,
            session=session,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            migration_request=migration_request,
            resp_cache=resp_cache,
            preserve_cache=preserve_cache,
            num_experts_per_tok=num_experts_per_tok,
            num_moe_layers=num_moe_layers,
        )

    def update_running(self, running: SeqList, batched_outputs: BatchedOutputs, is_decoding: bool) -> None:
        """Update running sequences."""
        next_token_ids = batched_outputs.next_token_ids
        stopped = batched_outputs.stopped
        stopped = stopped.tolist()
        model_metas = batched_outputs.model_metas
        if model_metas is None:
            model_metas = [None] * len(running)

        next_token_ids = next_token_ids.numpy()
        all_expert_ids = [None] * len(running)
        if is_decoding:
            num_tokens = [1] * len(running)
        else:
            num_tokens = [msg.num_token_ids for msg in running]

        if batched_outputs.extra_outputs.all_expert_ids is not None:
            all_expert_ids = batched_outputs.extra_outputs.all_expert_ids.split(num_tokens, dim=0)

        update_mode = UpdateTokenMode.DECODE if is_decoding else UpdateTokenMode.PREFILL
        for token, msg, stop, model_meta, expert_ids in zip(next_token_ids, running, stopped, model_metas,
                                                            all_expert_ids):
            if msg.status != MessageStatus.LOCKED:
                continue

            # fill token
            msg.update_token_ids(token, model_meta=model_meta, mode=update_mode, expert_ids=expert_ids)
            if stop:
                msg.status = MessageStatus.TO_BE_MIGRATED if msg.preserve_cache else MessageStatus.STOPPED
