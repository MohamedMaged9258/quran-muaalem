from dataclasses import dataclass

import torch


@dataclass
class ModelInput:
    input_features: torch.FloatTensor
    attention_mask: torch.LongTensor
