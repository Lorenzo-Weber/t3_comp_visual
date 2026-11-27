import random
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

SAMPLES_PER_FOLDER = 200
CROP = 128
BATCH = 32
EPOCHS = 1
LR = 1e-4

SIGMA_MIN = 0.7
SIGMA_MAX = 1.5
BM_SCALE = 10.0

T = 1000       # passos usados no treinamento da difusao
STEPS = 50     # passos usados na restauracao

SEEDS = [42]
SPLIT_SEED = 0
VAL_SEED = 1234
EVAL_SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

random.seed(SPLIT_SEED)
subfolders = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
selected_files = []

for folder in subfolders:
    images = sorted(folder.glob("*.png"))
    selected_files.extend(random.sample(images, SAMPLES_PER_FOLDER))

random.shuffle(selected_files)

n_train = int(0.8 * len(selected_files))
n_val = int(0.1 * len(selected_files))

train_files = selected_files[:n_train]
val_files = selected_files[n_train:n_train + n_val]
test_files = selected_files[n_train + n_val:]

print(f"Total de imagens: {len(selected_files)}")
print(f"Treino: {len(train_files)} | Validacao: {len(val_files)} | Teste: {len(test_files)}")


class BlurDataset(Dataset):
    def __init__(self, files, train=False):
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        width, height = img.size

        # no treino, o recorte e o blur variam. Na validacao/teste, sao fixos.
        rng = random if self.train else random.Random(index)
        generator = None if self.train else torch.Generator().manual_seed(index)

        if self.train:
            left = rng.randint(0, width - CROP)
            top = rng.randint(0, height - CROP)
        else:
            left = (width - CROP) // 2
            top = (height - CROP) // 2

        img = img.crop((left, top, left + CROP, top + CROP))

        sigma = rng.uniform(SIGMA_MIN, SIGMA_MAX)
        blurred = img.filter(ImageFilter.GaussianBlur(sigma))

        hq = to_tensor(img) * 2 - 1
        lq = to_tensor(blurred) * 2 - 1

        noise = torch.randn(lq.shape, generator=generator)
        noise_level = rng.uniform(0, 7)
        lq = lq + noise * (noise_level / 255.0 * 2)

        bm = torch.tensor(sigma / BM_SCALE, dtype=torch.float32)
        return hq, lq, bm


train_ds = BlurDataset(train_files, train=True)
val_ds = BlurDataset(val_files, train=False)
test_ds = BlurDataset(test_files, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device).float()
        / max(half - 1, 1)
    )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)

    if dim % 2:
        emb = F.pad(emb, (0, 1))

    return emb


class ResBlock(nn.Module):
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

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_ch, emb_ch * 2),
            nn.SiLU(),
            nn.Linear(emb_ch * 2, emb_ch),
        )

        self.bm_mlp = nn.Sequential(
            nn.Linear(1, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # entrada: imagem ruidosa da difusao + imagem borrada.
        self.input = nn.Conv2d(6, base, 3, padding=1)

        self.enc1 = nn.ModuleList([
            ResBlock(base, base, emb_ch),
            ResBlock(base, base, emb_ch),
        ])
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.enc2 = nn.ModuleList([
            ResBlock(base * 2, base * 2, emb_ch),
            ResBlock(base * 2, base * 2, emb_ch),
        ])
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.middle = nn.ModuleList([
            ResBlock(base * 4, base * 4, emb_ch),
            ResBlock(base * 4, base * 4, emb_ch),
        ])

        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2 = nn.ModuleList([
            ResBlock(base * 4, base * 2, emb_ch),
            ResBlock(base * 2, base * 2, emb_ch),
        ])

        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = nn.ModuleList([
            ResBlock(base * 2, base, emb_ch),
            ResBlock(base, base, emb_ch),
        ])

        self.output = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def forward(self, x_t, lq, t, bm):
        emb = (
            self.time_mlp(timestep_embedding(t))
            + self.bm_mlp(bm.reshape(-1, 1))
        )

        x = self.input(torch.cat([x_t, lq], dim=1))

        s1 = x
        for block in self.enc1:
            s1 = block(s1, emb)

        x = self.down1(s1)

        s2 = x
        for block in self.enc2:
            s2 = block(s2, emb)

        x = self.down2(s2)

        for block in self.middle:
            x = block(x, emb)

        x = F.interpolate(x, size=s2.shape[-2:], mode="nearest")
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        for block in self.dec2:
            x = block(x, emb)

        x = F.interpolate(x, size=s1.shape[-2:], mode="nearest")
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        for block in self.dec1:
            x = block(x, emb)

        return self.output(x)
    
class SuperSharpenMobile(nn.Module):
    STAGES = [
        (0, 2, 16, 2),
        (2, 4, 24, 4),
        (4, 7, 32, 8),
        (7, 14, 96, 16),
        (14, 18, 320, 32),
    ]

    def __init__(self, ch=(32, 48, 64, 96, 128, 128), emb_ch=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_ch, emb_ch * 2),
            nn.SiLU(),
            nn.Linear(emb_ch * 2, emb_ch),
        )

        self.bm_mlp = nn.Sequential(
            nn.Linear(1, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        features = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.enc = nn.ModuleList([
            features[start:end] for start, end, _, _ in self.STAGES
        ])

        # Normalizacao usada no pre-treinamento do ImageNet.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Stem em resolucao total para preservar detalhes finos.
        self.input = nn.Conv2d(6, ch[0], 3, padding=1)
        self.stem = nn.ModuleList([
            ResBlock(ch[0], ch[0], emb_ch),
            ResBlock(ch[0], ch[0], emb_ch),
        ])

        self.proj = nn.ModuleList([
            nn.Conv2d(channels + 3, ch[k + 1], 1)
            for k, (_, _, channels, _) in enumerate(self.STAGES)
        ])

        self.middle = nn.ModuleList([
            ResBlock(ch[5], ch[5], emb_ch)
        ])

        self.ups = nn.ModuleList([
            nn.Conv2d(ch[k + 1], ch[k], 3, padding=1)
            for k in range(5)
        ])

        n_blocks = [2, 2, 1, 1, 1]
        self.decs = nn.ModuleList([
            nn.ModuleList(
                [ResBlock(ch[k] * 2, ch[k], emb_ch)]
                + [ResBlock(ch[k], ch[k], emb_ch) for _ in range(n_blocks[k] - 1)]
            )
            for k in range(5)
        ])

        self.output = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 3, 3, padding=1),
        )

    def forward(self, x_t, lq, t, bm):
        emb = (
            self.time_mlp(timestep_embedding(t))
            + self.bm_mlp(bm.reshape(-1, 1))
        )

        z = ((lq + 1) / 2 - self.mean) / self.std
        features = []
        for stage in self.enc:
            z = stage(z)
            features.append(z)

        s0 = self.input(torch.cat([x_t, lq], dim=1))
        for block in self.stem:
            s0 = block(s0, emb)

        skips = [s0]
        for k, feature in enumerate(features):
            stride = self.STAGES[k][3]
            x_t_scaled = F.avg_pool2d(x_t, stride)
            skips.append(self.proj[k](torch.cat([feature, x_t_scaled], dim=1)))

        x = skips[5]
        for block in self.middle:
            x = block(x, emb)

        for k in range(4, -1, -1):
            x = F.interpolate(x, size=skips[k].shape[-2:], mode="nearest")
            x = self.ups[k](x)
            x = torch.cat([x, skips[k]], dim=1)

            for block in self.decs[k]:
                x = block(x, emb)

        return self.output(x)

betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
alpha_bar = torch.cumprod(1 - betas, dim=0)


def calculate_loss(model, hq, lq, bm):
    t = torch.randint(0, T, (hq.shape[0],), device=DEVICE)
    noise = torch.randn_like(hq)

    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = alpha_bar_t.sqrt() * hq + (1 - alpha_bar_t).sqrt() * noise

    predicted_noise = model(x_t, lq, t, bm)
    return F.mse_loss(predicted_noise, noise)


def train_model(model, optimizer, name, seed, history):
    checkpoint_path = CKPT_DIR / f"{name}_seed{seed}.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    start_time = time.time()

    # Reinicia a aleatoriedade para comparar os modelos com as mesmas seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0

        for hq, lq, bm in train_loader:
            hq = hq.to(DEVICE)
            lq = lq.to(DEVICE)
            bm = bm.to(DEVICE)

            loss = calculate_loss(model, hq, lq, bm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item()

        train_loss = train_total / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_total = 0.0

        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(VAL_SEED)
            for hq, lq, bm in val_loader:
                hq = hq.to(DEVICE)
                lq = lq.to(DEVICE)
                bm = bm.to(DEVICE)
                val_total += calculate_loss(model, hq, lq, bm).item()

        val_loss = val_total / len(val_loader)
        val_losses.append(val_loss)

        history.append({
            "model": name,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        pd.DataFrame(history).to_csv("curvas_treinamento.csv", index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"[{name} | seed {seed}] epoca {epoch + 1}/{EPOCHS} | "
            f"treino: {train_loss:.5f} | validacao: {val_loss:.5f}"
        )

    return {
        "checkpoint": checkpoint_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "minutes": (time.time() - start_time) / 60,
    }


@torch.no_grad()
def sample(model, lq, bm):
    x = torch.randn_like(lq)
    timesteps = torch.linspace(T - 1, 0, STEPS).long().tolist()

    for i, timestep in enumerate(timesteps):
        t_batch = torch.full(
            (lq.shape[0],), timestep, device=DEVICE, dtype=torch.long
        )
        predicted_noise = model(x, lq, t_batch, bm)

        alpha_bar_t = alpha_bar[timestep]
        if i + 1 < len(timesteps):
            alpha_bar_previous = alpha_bar[timesteps[i + 1]]
        else:
            alpha_bar_previous = torch.tensor(1.0, device=DEVICE)

        predicted_clean = (
            (x - (1 - alpha_bar_t).sqrt() * predicted_noise)
            / alpha_bar_t.sqrt()
        ).clamp(-1, 1)

        x = (
            alpha_bar_previous.sqrt() * predicted_clean
            + (1 - alpha_bar_previous).sqrt() * predicted_noise
        )

    return x

def tensor_to_image(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def calculate_psnr(reference, image):
    mse = np.mean((reference - image) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def calculate_ssim(reference, image):
    return structural_similarity(
        reference, image, channel_axis=2, data_range=1.0
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_blurred, psnr_restored = [], []
    ssim_blurred, ssim_restored = [], []

    with torch.random.fork_rng():
        torch.manual_seed(EVAL_SEED)

        for hq, lq, bm in loader:
            hq = hq.to(DEVICE)
            lq = lq.to(DEVICE)
            bm = bm.to(DEVICE)

            restored = sample(model, lq, bm)

            for i in range(hq.shape[0]):
                reference = tensor_to_image(hq[i])
                blurred = tensor_to_image(lq[i])
                output = tensor_to_image(restored[i])

                psnr_blurred.append(calculate_psnr(reference, blurred))
                psnr_restored.append(calculate_psnr(reference, output))
                ssim_blurred.append(calculate_ssim(reference, blurred))
                ssim_restored.append(calculate_ssim(reference, output))

    return {
        "psnr_lq": float(np.mean(psnr_blurred)),
        "psnr_out": float(np.mean(psnr_restored)),
        "ssim_lq": float(np.mean(ssim_blurred)),
        "ssim_out": float(np.mean(ssim_restored)),
    }

results = []
history = []

for seed in SEEDS:

    torch.manual_seed(seed)
    lite_model = SuperSharpenLite().to(DEVICE)
    lite_optimizer = torch.optim.AdamW(lite_model.parameters(), lr=LR)

    print(f"\n===== SuperSharpenLite | seed {seed} =====")
    lite_training = train_model(
        lite_model, lite_optimizer, "supersharpen_lite", seed, history
    )

    lite_model.load_state_dict(torch.load(lite_training["checkpoint"], map_location=DEVICE))
    lite_metrics = evaluate(lite_model, test_loader)

    results.append({
        "model": "supersharpen_lite",
        "seed": seed,
        "best_val_loss": lite_training["best_val_loss"],
        "best_epoch": lite_training["best_epoch"],
        "minutes": lite_training["minutes"],
        **lite_metrics,
    })
    pd.DataFrame(results).to_csv("resultados_comparacao.csv", index=False)

    print(
        f"SuperSharpenLite | PSNR: {lite_metrics['psnr_lq']:.2f} -> "
        f"{lite_metrics['psnr_out']:.2f} dB | SSIM: "
        f"{lite_metrics['ssim_lq']:.4f} -> {lite_metrics['ssim_out']:.4f}"
    )

    del lite_model, lite_optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    mobile_model = SuperSharpenMobile().to(DEVICE)

    mobile_encoder_params = list(mobile_model.enc.parameters())
    mobile_other_params = [
        p for name, p in mobile_model.named_parameters()
        if not name.startswith("enc.")
    ]

    mobile_optimizer = torch.optim.AdamW([
        {"params": mobile_encoder_params, "lr": LR * 0.1},
        {"params": mobile_other_params, "lr": LR},
    ])

    print(f"\n===== MobileNetV2 pre-treinada | seed {seed} =====")
    mobile_training = train_model(
        mobile_model, mobile_optimizer, "mobilenetv2_pretrained", seed, history
    )

    mobile_model.load_state_dict(torch.load(mobile_training["checkpoint"], map_location=DEVICE))
    mobile_metrics = evaluate(mobile_model, test_loader)

    results.append({
        "model": "mobilenetv2_pretrained",
        "seed": seed,
        "best_val_loss": mobile_training["best_val_loss"],
        "best_epoch": mobile_training["best_epoch"],
        "minutes": mobile_training["minutes"],
        **mobile_metrics,
    })
    pd.DataFrame(results).to_csv("resultados_comparacao.csv", index=False)

    print(
        f"MobileNetV2 pre-treinada | PSNR: {mobile_metrics['psnr_lq']:.2f} -> "
        f"{mobile_metrics['psnr_out']:.2f} dB | SSIM: "
        f"{mobile_metrics['ssim_lq']:.4f} -> {mobile_metrics['ssim_out']:.4f}"
    )

    del mobile_model, mobile_optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
summary = results_df.groupby("model", sort=False).agg(
    psnr_mean=("psnr_out", "mean"),
    psnr_std=("psnr_out", "std"),
    ssim_mean=("ssim_out", "mean"),
    ssim_std=("ssim_out", "std"),
    val_loss_mean=("best_val_loss", "mean"),
    best_epoch_mean=("best_epoch", "mean"),
    minutes_mean=("minutes", "mean"),
)
summary.to_csv("resumo_comparacao.csv")

print("\n===== RESUMO DA COMPARACAO =====")
print(summary.round(4).to_string())

lite_results = results_df[results_df["model"] == "supersharpen_lite"].sort_values("seed")
mobile_results = results_df[results_df["model"] == "mobilenetv2_pretrained"].sort_values("seed")

psnr_difference = mobile_results["psnr_out"].to_numpy() - lite_results["psnr_out"].to_numpy()
ssim_difference = mobile_results["ssim_out"].to_numpy() - lite_results["ssim_out"].to_numpy()

print("\n===== DIFERENCA: MOBILENETV2 - SUPERSHARPEN LITE =====")
print(f"PSNR: media = {psnr_difference.mean():.4f} | desvio = {psnr_difference.std(ddof=1):.4f}")
print(f"SSIM: media = {ssim_difference.mean():.4f} | desvio = {ssim_difference.std(ddof=1):.4f}")

lite_model = SuperSharpenLite().to(DEVICE)
lite_model.load_state_dict(torch.load(CKPT_DIR / f"supersharpen_lite_seed{SEEDS[0]}.pt", map_location=DEVICE))
lite_model.eval()

mobile_model = SuperSharpenMobile().to(DEVICE)
mobile_model.load_state_dict(torch.load(CKPT_DIR / f"mobilenetv2_pretrained_seed{SEEDS[0]}.pt", map_location=DEVICE))
mobile_model.eval()

hq, lq, bm = next(iter(test_loader))
hq = hq.to(DEVICE)
lq = lq.to(DEVICE)
bm = bm.to(DEVICE)

with torch.random.fork_rng():
    torch.manual_seed(EVAL_SEED)
    lite_output = sample(lite_model, lq, bm)

with torch.random.fork_rng():
    torch.manual_seed(EVAL_SEED)
    mobile_output = sample(mobile_model, lq, bm)

columns = ["LQ", "SuperSharpenLite", "MobileNetV2 pre-treinada", "HQ"]
n_rows = min(4, hq.shape[0])
fig, axes = plt.subplots(n_rows, len(columns), figsize=(12, 3 * n_rows))

for i in range(n_rows):
    images = [lq[i], lite_output[i], mobile_output[i], hq[i]]
    for ax, image, title in zip(axes[i], images, columns):
        ax.imshow(tensor_to_image(image))
        ax.set_title(title)
        ax.axis("off")

plt.tight_layout()
plt.savefig("comparacao_visual.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nResultados salvos:")
print("- resultados_comparacao.csv: metricas de cada modelo e seed")
print("- resumo_comparacao.csv: media e desvio-padrao das metricas")
print("- curvas_treinamento.csv: loss de treino e validacao por epoca")
print("- comparacao_visual.png: exemplos das imagens restauradas")
print(f"- {CKPT_DIR}/: melhor checkpoint de cada modelo em cada seed")
