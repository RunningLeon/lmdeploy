# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear,
                                        build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class InternLM3SelfAttention(nn.Module):
    """Rewrite module of InternLM3SelfAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        # packed qkv
        self.wqkv = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=config.bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
        )

        # o_proj
        self.wo = build_rowwise_linear(num_heads * head_dim,
                                       hidden_size,
                                       bias=config.bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of InternLM2Attention.forward."""
        # qkv proj
        qkv_states = self.wqkv(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.wqkv.split_qkv(
            qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.wo(attn_output)
        return attn_output


class InternLM3CrossAttention(nn.Module):
    """Rewrite module of InternLM3CrossAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        # wq
        self.wq = build_colwise_linear(hidden_size,
                                       num_heads * head_dim,
                                       bias=config.bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
        )

        # o_proj
        self.wo = build_rowwise_linear(num_heads * head_dim,
                                       hidden_size,
                                       bias=config.bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                attn_metadata: Any = None,
                key_states: Optional[torch.Tensor] = None,
                value_states: Optional[torch.Tensor] = None):
        """Rewrite of InternLM2Attention.forward."""
        # q proj
        query_states = self.wq(hidden_states)
        # (-1, heads, head_dim)
        query_states = query_states.flatten(0, -2)
        query_states = query_states.reshape(query_states.shape[0], -1,
                                            self.head_dim)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )
        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.wo(attn_output)
        return attn_output


class InternLM3MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.w2 = build_rowwise_linear(config.intermediate_size,
                                       config.hidden_size,
                                       bias=False,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.w2(act)


class InternLM3DecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_cross_decoder: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_cross_decoder = is_cross_decoder
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        if is_cross_decoder:
            self.attention = InternLM3CrossAttention(config,
                                                     dtype=dtype,
                                                     device=device)
        else:
            self.attention = InternLM3SelfAttention(config,
                                                    dtype=dtype,
                                                    device=device)

        # builf MLP
        self.feed_forward = InternLM3MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.attention_norm = RMSNorm(config.hidden_size,
                                      config.rms_norm_eps,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device)

        # build attention layer norm
        self.ffn_norm = RMSNorm(config.hidden_size,
                                config.rms_norm_eps,
                                quant_config=quantization_config,
                                dtype=dtype,
                                device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                past_key_value: Optional[List[torch.FloatTensor]],
                residual: Optional[torch.Tensor] = None,
                attn_metadata: Any = None,
                key_states: Optional[torch.Tensor] = None,
                value_states: Optional[torch.Tensor] = None):

        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)

        # Self Attention
        attn_kw_args = dict(hidden_states=hidden_states,
                            rotary_pos_emb=rotary_pos_emb,
                            past_key_value=past_key_value,
                            attn_metadata=attn_metadata)

        if self.is_cross_decoder:
            attn_kw_args.update(
                dict(key_states=key_states, value_states=value_states))
        hidden_states = self.attention(**attn_kw_args)

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class InternLM3SelfDecoder(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            InternLM3DecoderLayer(config, i, dtype, device)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )
        return hidden_states, residual


class InternLM3CrossDecoder(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config

        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        self.wk = build_colwise_linear(hidden_size,
                                       num_key_value_heads * head_dim,
                                       bias=config.bias,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)
        self.wv = build_colwise_linear(hidden_size,
                                       num_key_value_heads * head_dim,
                                       bias=config.bias,
                                       dtype=dtype,
                                       device=device,
                                       is_tp=True)
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        self.layers = nn.ModuleList([
            InternLM3DecoderLayer(config,
                                  i,
                                  dtype,
                                  device,
                                  is_cross_decoder=True)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        hidden_states_norm, residual = self.norm(hidden_states, residual)
        key_states = self.wk(hidden_states_norm).flatten(0, -2)
        value_states = self.wv(hidden_states_norm).flatten(0, -2)
        seq_len = key_states.shape[0]
        key_states = key_states.reshape(seq_len, -1, self.head_dim)
        value_states = value_states.reshape(seq_len, -1, self.head_dim)
        shared_kv_cache = past_key_values[-1]

        hidden_states, residual = self.layers[0](
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=shared_kv_cache,
            residual=residual,
            attn_metadata=attn_metadata,
            key_states=key_states,
            value_states=value_states)

        pre_hidden_states = None
        pre_residual = None
        if not attn_metadata.is_decoding:
            # only let last tokens to compute in prefill stage
            # so we slice out the hidden states of last tokens
            last_token_indices = attn_metadata.last_token_indices
            pre_hidden_states = hidden_states
            pre_residual = residual
            hidden_states = hidden_states.index_select(1, last_token_indices)
            residual = residual.index_select(1, last_token_indices)
            rotary_pos_emb = [
                _.index_select(0, last_token_indices) for _ in rotary_pos_emb
            ]
            attn_metadata.q_start_loc = attn_metadata.new_q_start_loc
            attn_metadata.q_seqlens = attn_metadata.new_q_seqlens

        for decoder_layer in self.layers[1:]:
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=shared_kv_cache,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if pre_hidden_states is not None:
            # assign back hidden states of last tokens
            last_token_indices = attn_metadata.last_token_indices
            pre_hidden_states[:,
                              last_token_indices, :] = hidden_states.squeeze(1)
            pre_residual[:, last_token_indices, :] = residual.squeeze(1)
            hidden_states = pre_hidden_states
            residual = pre_residual
        return hidden_states, residual


class InternLM3Model(nn.Module):
    """internlm3 model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        quantization_config = getattr(config, 'quantization_config', None)

        self.tok_embeddings = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           self.padding_idx,
                                           dtype=dtype,
                                           device=device)

        # build all decode layers
        self.self_decoder = InternLM3SelfDecoder(config,
                                                 dtype=dtype,
                                                 device=device)
        self.cross_decoder = InternLM3CrossDecoder(config,
                                                   dtype=dtype,
                                                   device=device)
        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build rotary embedding in Model
        rope_scaling = config.rope_scaling
        scaling_factor = 1.0
        emb_type = RopeType.LinearScaling
        if rope_scaling is not None:
            scaling_factor = rope_scaling.get('factor', scaling_factor)
            rope_type = rope_scaling['type']
            if rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            if rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            emb_type=emb_type,
        )

    def _update_attn_metadata(self, attn_metadata: Any):
        """update attn meatada for cross attn layers in prefill stage."""
        n_seqs = attn_metadata.q_seqlens.shape[0]
        last_token_indices = attn_metadata.q_seqlens - 1
        new_q_start_loc = torch.arange(n_seqs,
                                       device=attn_metadata.q_seqlens.device)
        new_q_seqlens = attn_metadata.q_seqlens * 0 + 1
        attn_metadata.new_q_start_loc = new_q_start_loc
        attn_metadata.new_q_seqlens = new_q_seqlens
        attn_metadata.last_token_indices = last_token_indices
        return attn_metadata

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of forward."""
        if not attn_metadata.is_decoding:
            attn_metadata = self._update_attn_metadata(attn_metadata)

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        hidden_states, residual = self.self_decoder(
            hidden_states,
            rotary_pos_emb,
            past_key_values=past_key_values,
            residual=residual,
            attn_metadata=attn_metadata)

        hidden_states, residual = self.cross_decoder(
            hidden_states,
            rotary_pos_emb,
            past_key_values=past_key_values,
            residual=residual,
            attn_metadata=attn_metadata)
        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.tok_embeddings


class InternLM3ForCausalLM(nn.Module, CudaGraphMixin):
    """rewrote model of InternLM3ForCausalLM."""

    packed_modules_mapping = {
        'gate_up_proj': [
            'w1',
            'w3',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build Model
        self.model = InternLM3Model(config, dtype=dtype, device=device)
        # build lm_head
        self.output = build_rowwise_linear(config.hidden_size,
                                           config.vocab_size,
                                           bias=False,
                                           dtype=dtype,
                                           device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.output(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """support cudagraph."""
        # disable cudagraph in prefill stage
        if not kwargs['attn_metadata'].is_decoding:
            return False

        seq_lens = input_ids.size(1)
        if seq_lens <= 512:
            return True
        return False

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    attn_metadata=attn_metadata,
                    inputs_embeds=inputs_embeds)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.w1', 0),
            ('.gate_up_proj', '.w3', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if '.wqkv' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight, layout='hgd')
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
