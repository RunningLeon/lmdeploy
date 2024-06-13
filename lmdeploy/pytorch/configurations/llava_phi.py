# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class LlavaPhiModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'LlavaPhiForCausalLM'

    @classmethod
    def build(cls, hf_config):
        """build llava phi."""
        hidden_size = hf_config.n_embd
        num_attention_heads = hf_config.n_head
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = hf_config.n_head_kv or num_attention_heads
        bos_token_id = 50256
        eos_token_id = 50256
        return ModelConfig(hidden_size=hidden_size,
                           num_layers=hf_config.n_layer,
                           num_attention_heads=num_attention_heads,
                           num_key_value_heads=num_key_value_heads,
                           bos_token_id=bos_token_id,
                           eos_token_id=eos_token_id,
                           head_dim=head_dim,
                           vocab_size=hf_config.vocab_size,
                           unused_modules=[
                               'lm_head', 'transformer.vision_tower',
                               'transformer.mm_projector'
                           ])
