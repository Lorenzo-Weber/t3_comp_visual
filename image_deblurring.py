import torch
import random
import math

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
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

# COMO VAI SER USADO NOTEBOOKS DA PRA SO CUSPIR CODIGO AQ E DEPOIS PORTAR, PQ ISSO TA HORRIVEL

def timestep_embedding(t, dim=128): # esses modelos de difusao precisam destes timestamps para saber em qual passo estao
    half = dim // 2

    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, device=t.device).float() /
        max(half - 1, 1)
    )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([args.sin(), args.cos()], dim=1)

    if dim % 2:
        emb = F.pad(emb, (0, 1))

    return emb


class ResBlock(nn.Module): # Bloco residual q concatena algumas camadas e dps soma com o X (skip conectipn), por isso usa h ao inves do proprio X
    def __init__(self, in_ch, out_ch, emb_ch=128):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.emb = nn.Linear(emb_ch, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)


class SuperSharpenLite(nn.Module):
    def __init__(self, base=32, emb_ch=128):
        super().__init__()

        # Conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_ch, emb_ch * 2),
            nn.SiLU(),
            nn.Linear(emb_ch * 2, emb_ch)
        )

        self.bm_mlp = nn.Sequential(
            nn.Linear(1, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        # Input: noisy image + degraded image
        self.input = nn.Conv2d(6, base, 3, padding=1)

        # Encoder
        self.enc1 = nn.ModuleList([
            ResBlock(base, base, emb_ch),
            ResBlock(base, base, emb_ch)
        ])

        self.down1 = nn.Conv2d(
            base, base * 2, 4, stride=2, padding=1
        )

        self.enc2 = nn.ModuleList([
            ResBlock(base * 2, base * 2, emb_ch),
            ResBlock(base * 2, base * 2, emb_ch)
        ])

        self.down2 = nn.Conv2d(
            base * 2, base * 4, 4, stride=2, padding=1
        )

        # Bottleneck
        self.middle = nn.ModuleList([
            ResBlock(base * 4, base * 4, emb_ch),
            ResBlock(base * 4, base * 4, emb_ch)
        ])

        # Decoder
        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)

        self.dec2 = nn.ModuleList([
            ResBlock(base * 4, base * 2, emb_ch),
            ResBlock(base * 2, base * 2, emb_ch)
        ])

        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)

        self.dec1 = nn.ModuleList([
            ResBlock(base * 2, base, emb_ch),
            ResBlock(base, base, emb_ch)
        ])

        self.output = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1)
        )

    def forward(self, x_t, lq, t, bm):

        emb = (
            self.time_mlp(timestep_embedding(t))
            + self.bm_mlp(bm.reshape(-1, 1))
        )

        # Encoder
        x = self.input(torch.cat([x_t, lq], dim=1))

        s1 = x
        for block in self.enc1:
            s1 = block(s1, emb)

        x = self.down1(s1)

        s2 = x
        for block in self.enc2:
            s2 = block(s2, emb)

        x = self.down2(s2)

        # Bottleneck
        for block in self.middle:
            x = block(x, emb)

        # Decoder
        x = F.interpolate(
            x, size=s2.shape[-2:], mode="nearest"
        )
        x = self.up2(x)

        x = torch.cat([x, s2], dim=1)

        for block in self.dec2:
            x = block(x, emb)

        x = F.interpolate(
            x, size=s1.shape[-2:], mode="nearest"
        )
        x = self.up1(x)

        x = torch.cat([x, s1], dim=1)

        for block in self.dec1:
            x = block(x, emb)

        return self.output(x)

# O MODELO N EH O SUPERSHARPEN PQ OS CARAS SIMPLESMENTE TIVERAM ACESSO A 4 H100 E 13 DIAS DE TREINO 

# TO DO:
# 1- ENTENDER MELHOR O PROCESSO DE BLUR (GAUSSIAN EH UMA OPCAO)
# 2- VERIFICAR O LOOP DE TREINO
# 3- IMPLEMENTAR TUDO ISSO AE
# 4- VER COMO FUNCIONA A INFERENCIA
# 5- TESTAR