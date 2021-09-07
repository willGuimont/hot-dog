import base64
from io import BytesIO

import requests
import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pipeline.pipeline as pp
from models.data_modules import SeeFoodDataModule
from models.dognet import DogNet

app = FastAPI()

origins = [
    'https://willguimont.gihub.io',
    'http://localhost',
    'http://localhost:5000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

checkpoint = BytesIO()
r = requests.get('https://gitlab.com/william.guimont-martin/hotdog-notahotdog/-/raw/main/model.ckpt')
for chunk in r.iter_content(chunk_size=512 * 1024):
    if chunk:
        checkpoint.write(chunk)
checkpoint.seek(0, 0)

input_size = 224
model = DogNet()
model.load_from_checkpoint(checkpoint)
transform = pp.Compose([
    T.Scale(input_size),
    T.RandomCrop(input_size),
    T.ToTensor(),
    T.Normalize(SeeFoodDataModule.mean, SeeFoodDataModule.std),
])


class ImageRequest(BaseModel):
    img: str


@app.get("/health")
async def health():
    return "ok!"


@app.post("/predict")
async def create_files(base64_images: ImageRequest):
    imgdata = base64.b64decode(base64_images.img)
    img_file = BytesIO(imgdata)
    img = Image.open(img_file).convert('RGB')

    x = transform(img)[None, :, :, :]
    y = model(x)
    is_hot_dog = torch.argmax(y, dim=1).item() > 0

    return {"isHotDog": is_hot_dog}
