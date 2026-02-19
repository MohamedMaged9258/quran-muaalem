from pandas.io.pytables import performance_doc
import os

import torch
from ray import serve
from ..modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .types import ModelInput
from time import perf_counter


@serve.deployment(
    name="model",
    ray_actor_options={"num_gpus": 1},
    num_replicas=1,
    max_ongoing_requests=128,
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

    @serve.batch(
        max_batch_size=32,
        batch_wait_timeout_s=2,
    )
    async def predict(
        self,
        model_inputs: list[ModelInput],
    ) -> list[dict[str, torch.Tensor]]:

        with torch.inference_mode():
            start = perf_counter()
            print("Batch Size")
            print(len(model_inputs))
            input_features = torch.cat([i.input_features for i in model_inputs]).to(
                self.device, dtype=self.dtype
            )
            attention_mask = torch.cat([i.attention_mask for i in model_inputs]).to(
                self.device, dtype=self.dtype
            )

            level_to_logits = self.model(
                input_features, attention_mask, return_dict=False
            )[0]

            list_of_level_to_logits = []
            d = {}
            for idx in range(level_to_logits["phonemes"].shape[0]):
                for level in level_to_logits:
                    d[level] = (
                        level_to_logits[level][idx]
                        .cpu()
                        .to(dtype=torch.float32)
                        .unsqueeze(0)
                    )
                list_of_level_to_logits.append(d)
            end = perf_counter()
            print(f"Model Time: {end - start}")
            return list_of_level_to_logits

    async def __call__(
        self,
        model_input: ModelInput,
    ) -> dict[str, torch.Tensor]:
        return await self.predict(model_input)
