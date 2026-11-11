"""Load the original InternVL2 remote model with Transformers 5 initialization."""

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def load_internvl2_model(variant):
    config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], variant)

    class TrainingInternVL(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The original remote constructor omits post_init. Transformers 5
            # needs its tied-weight and loading metadata before loading weights.
            self.post_init()

    # Transformers 5 defaults to the checkpoint's BF16 storage dtype. Keep FP32
    # training parameters; Lightning applies mixed precision to computations.
    return TrainingInternVL.from_pretrained(
        variant, config=config, dtype=torch.float32, use_flash_attn=False
    )
