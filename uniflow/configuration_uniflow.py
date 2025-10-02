import os
from typing import Optional, Tuple, Union
from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)



class UniFlowVisionConfig(PretrainedConfig):
    model_type = 'uniflow'

    def __init__(
            self,
            num_channels=3,
            patch_size=14,
            image_size=224,
            qkv_bias=False,
            hidden_size=3200,
            num_attention_heads=25,
            intermediate_size=12800,
            qk_normalization=True,
            num_hidden_layers=48,
            use_flash_attn=True,
            hidden_act='gelu',
            norm_type='rms_norm',
            layer_norm_eps=1e-6,
            dropout=0.0,
            drop_path_rate=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=0.1,
            # enc_proj
            vit_hidden_size=1024,
            llm_hidden_size=1536,
            latent_ch=64,
            # flow decoder
            use_global_blocks=True,
            global_blocks_depth=6,
            num_decoder_layers=12,
            num_sampling_steps='100',
            use_disp_loss=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn
        # enc_proj
        self.vit_hidden_size = vit_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.latent_ch = latent_ch
        self.use_disp_loss = use_disp_loss
        # decoder
        self.global_blocks_depth = global_blocks_depth
        self.num_decoder_layers = num_decoder_layers
        self.num_sampling_steps = num_sampling_steps
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)

