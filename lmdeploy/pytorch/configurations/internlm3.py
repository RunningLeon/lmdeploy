# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class InternLM3ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'InternLM3'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config)
        cfg.num_layers = hf_config.num_hidden_layers + 1
        return cfg
