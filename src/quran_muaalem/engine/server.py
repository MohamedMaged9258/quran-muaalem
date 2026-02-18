from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch

from .preprocessing import PreprocessingDeployment
from .model import ModelDeployment
from .postprocessing import PostProcessingDeployment

app = FastAPI(title="Quran Muaalem API")


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class QuranMuaalemAPI:
    def __init__(
        self,
        preprocessing: DeploymentHandle,
        model: DeploymentHandle,
        postprocessing: DeploymentHandle,
    ):
        self.preprocessing = preprocessing
        self.model = model
        self.postprocessing = postprocessing

    @app.post("/predict")
    async def predict(self, audio: UploadFile = File(...)):
        try:
            audio_bytes = await audio.read()

            model_input = await self.preprocessing.remote(audio_bytes)

            logits = await self.model.remote(model_input)

            result = await self.postprocessing.remote(logits)

            return JSONResponse(content=result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health(self):
        return {"status": "healthy"}


muaalem_app = QuranMuaalemAPI.bind(
    preprocessing=PreprocessingDeployment.bind(),
    model=ModelDeployment.bind(),
    postprocessing=PostProcessingDeployment.bind(),
)
