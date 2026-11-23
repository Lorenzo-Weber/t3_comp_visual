import torch
import random
import math

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from PIL import ImageFilter

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data")
SUB_DIRS = sorted([folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()])

SAMPLES_PER_FOLDER = 300
NUM_FOLDERS = len(SUB_DIRS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP = 128            
BATCH = 32
EPOCHS = 250
LR = 1e-4

SIGMA_MIN = 0.7       # faixa de blur usada no teste sintetico do paper
SIGMA_MAX = 7.5
BM_SCALE = 10.0       # divido o sigma por 10 pro BM ficar entre 0 e 1 (o sigma maximo do paper eh 10)

T = 1000              # passos da difusao eh aquele t do timestamp
STEPS = 50            # passos da inferencia

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

# dicionario nome -> caminho
paths = {p.name: p for p in DATA_DIR.rglob("*.png")}

filenames = df["filename"].tolist()
random.shuffle(filenames)

n_train = int(0.8 * len(filenames)) # 80% teste, 10 validacao e 10 teste
n_val = int(0.1 * len(filenames))

train_files = filenames[:n_train]
val_files = filenames[n_train:n_train + n_val]
test_files = filenames[n_train + n_val:]

print(len(train_files), len(val_files), len(test_files))

class BlurDataset(Dataset):
    def __init__(self, files, train):
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(paths[self.files[i]]).convert("RGB")
        w, h = img.size  # assume que toda imagem eh maior que o CROP

        # treino: recorte aleatorio / val e teste: recorte no centro
        if self.train:
            left = random.randint(0, w - CROP)
            top = random.randint(0, h - CROP)
        else:
            left = (w - CROP) // 2
            top = (h - CROP) // 2

        img = img.crop((left, top, left + CROP, top + CROP))

        # o blur gaussiano do PIL usa o raio como se fosse o proprio sigma
        sigma = random.uniform(SIGMA_MIN, SIGMA_MAX)
        lq = img.filter(ImageFilter.GaussianBlur(sigma))

        # passa pra tensor e leva de [0, 1] pra [-1, 1]
        hq = to_tensor(img) * 2 - 1
        lq = to_tensor(lq) * 2 - 1

        # ruido gaussiano de 0 a 7 (escala 0-255), vezes 2 pq a imagem agora vai de -1 a 1
        lq = lq + torch.randn_like(lq) * random.uniform(0, 7) / 255 * 2

        # o blur measure de treino eh o proprio sigma aplicado
        bm = torch.tensor(sigma / BM_SCALE, dtype=torch.float32)

        return hq, lq, bm


train_ds = BlurDataset(train_files, train=True)
val_ds = BlurDataset(val_files, train=False)
test_ds = BlurDataset(test_files, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
test_loader = DataLoader(test_ds, batch_size=BATCH)

betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alpha_bar = torch.cumprod(1 - betas, dim=0)  # alpha_bar[t] = produto dos (1 - beta) ate t

model = SuperSharpenLite().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def calcula_loss(model, hq, lq, bm):
    # sorteia um t pra cada imagem do batch
    t = torch.randint(0, T, (hq.shape[0],), device=DEVICE)

    # forward da difusao: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * ruido
    eps = torch.randn_like(hq)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = ab.sqrt() * hq + (1 - ab).sqrt() * eps

    # tenta adivinhar o ruido que foi colocado
    pred = model(x_t, lq, t, bm)

    return F.mse_loss(pred, eps)


train_losses = []
val_losses = []
best_val = float("inf")

for epoch in range(EPOCHS):

    model.train()
    total = 0

    for hq, lq, bm in train_loader:
        hq, lq, bm = hq.to(DEVICE), lq.to(DEVICE), bm.to(DEVICE)

        loss = calcula_loss(model, hq, lq, bm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    train_losses.append(total / len(train_loader))

    # validacao (sem gradiente)
    model.eval()
    total = 0

    with torch.no_grad():
        for hq, lq, bm in val_loader:
            hq, lq, bm = hq.to(DEVICE), lq.to(DEVICE), bm.to(DEVICE)
            total += calcula_loss(model, hq, lq, bm).item()

    val_losses.append(total / len(val_loader))

    # guarda so o melhor modelo da validacao
    if val_losses[-1] < best_val:
        best_val = val_losses[-1]
        torch.save(model.state_dict(), "melhor_modelo.pt")

    print(f"epoca {epoch + 1}/{EPOCHS} | treino {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")

plt.plot(train_losses, label="treino")
plt.plot(val_losses, label="validacao")
plt.xlabel("epoca")
plt.ylabel("MSE do ruido")
plt.legend()
plt.show()

@torch.no_grad()
def sample(model, lq, bm):
    # bm ja tem que vir normalizado 
    x = torch.randn_like(lq)  # comeca de ruido puro

    # lista de timesteps indo de T-1 ate 0
    ts = torch.linspace(T - 1, 0, STEPS).long().tolist()

    for i, t in enumerate(ts):
        t_batch = torch.full((lq.shape[0],), t, device=DEVICE)

        # em todo passo o lq e o bm entram de novo na rede
        eps = model(x, lq, t_batch, bm)

        ab = alpha_bar[t]
        ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=DEVICE)

        # estima a imagem limpa e depois anda um passo pra tras
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps

    return x

model.load_state_dict(torch.load("melhor_modelo.pt"))
model.eval()

# seeds fixas pra o teste dar sempre o mesmo resultado
# DEPOIS TEM Q FAZER TESTES COM PELO MENOS 3 SEEDS
random.seed(0)
torch.manual_seed(0)


def para_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def psnr(a, b):
    return 10 * np.log10(1 / ((a - b) ** 2).mean())


def ssim(a, b):
    return structural_similarity(a, b, channel_axis=2, data_range=1.0)


psnr_lq, psnr_out, ssim_lq, ssim_out = [], [], [], []

for hq, lq, bm in test_loader:
    hq, lq, bm = hq.to(DEVICE), lq.to(DEVICE), bm.to(DEVICE)

    # aqui uso o blur real como BM, que eh o melhor caso
    out = sample(model, lq, bm)

    for i in range(hq.shape[0]):
        h, l, o = para_img(hq[i]), para_img(lq[i]), para_img(out[i])

        psnr_lq.append(psnr(h, l))
        psnr_out.append(psnr(h, o))
        ssim_lq.append(ssim(h, l))
        ssim_out.append(ssim(h, o))

# a imagem borrada de entrada serve de baseline pra comparar
print(f"PSNR  borrada: {np.mean(psnr_lq):.2f} | restaurada: {np.mean(psnr_out):.2f}")
print(f"SSIM  borrada: {np.mean(ssim_lq):.3f} | restaurada: {np.mean(ssim_out):.3f}")

hq, lq, bm = next(iter(test_loader))
hq, lq, bm = hq.to(DEVICE), lq.to(DEVICE), bm.to(DEVICE)
out = sample(model, lq, bm)

fig, axes = plt.subplots(4, 3, figsize=(9, 12))

for i in range(4):
    for ax, img, titulo in zip(axes[i], [lq[i], out[i], hq[i]], ["LQ", "Restaurada", "HQ"]):
        ax.imshow(para_img(img))
        ax.set_title(titulo)
        ax.axis("off")

plt.tight_layout()
plt.show()

hq, lq, bm = test_ds[0]
lq = lq.unsqueeze(0).to(DEVICE)

valores = [0.7, 2.0, 4.0, 6.0]

fig, axes = plt.subplots(1, len(valores) + 2, figsize=(18, 3.5))

axes[0].imshow(para_img(lq[0]))
axes[0].set_title("LQ")

for ax, v in zip(axes[1:], valores):
    bm_v = torch.full((1,), v / BM_SCALE, device=DEVICE)
    ax.imshow(para_img(sample(model, lq, bm_v)[0]))
    ax.set_title(f"BM={v}")

axes[-1].imshow(para_img(hq))
axes[-1].set_title("HQ")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()