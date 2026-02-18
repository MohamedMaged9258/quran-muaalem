import os

import torch
from ray import serve
from ..modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC


@serve.deployment(
    name="model",
    ray_actor_options={"num_gpus": 1},
)
class ModelDeployment:
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        dtype=torch.bfloat16,
    ):

        device_id = os.environ.get("cuda_visible_devices", "0").split(",")[0]
        self.device = f"cuda:{device_id}"
        self.dtype = dtype

        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
        self.model.to(self.device, dtype=dtype)
        self.model.eval()

    @torch.no_grad()
    # @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1) TODO:
    def __call__(
        self, input_features: torch.FloatTensor, attention_mask: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        input_features = input_features.to(self.device, dtype=self.dtype)
        attention_mask = attention_mask.to(self.device, dtype=self.dtype)

        level_to_logits = self.model(input_features, attention_mask, return_dict=False)[
            0
        ]

        return {v: g.cpu() for v, g in level_to_logits.items()}
