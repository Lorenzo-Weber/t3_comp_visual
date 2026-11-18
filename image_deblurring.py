import torch
import random

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

import torch.nn as nn
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
SUB_DIRS = sorted([folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()])

SAMPLES_PER_FOLDER = 200
NUM_FOLDERS = len(SUB_DIRS)

df = pd.DataFrame()

for folder in SUB_DIRS:
    PATH = DATA_DIR / folder

    images = list(PATH.glob("*.png"))
    selected = random.sample(images, SAMPLES_PER_FOLDER)

    dfAux = pd.DataFrame({
        "filename": [img.name for img in selected]
    })

    df = pd.concat([df, dfAux], ignore_index=True)

print(df)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

samples = random.sample(df["filename"].tolist(), 16)

for ax, filename in zip(axes.flatten(), samples):
    img_path = next(DATA_DIR.rglob(filename))
    img = Image.open(img_path)

    ax.imshow(img)
    ax.set_title(filename)
    ax.axis("off")

plt.tight_layout()
plt.show()

