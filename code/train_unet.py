#!/usr/bin/env python3
"""High-resolution semantic segmentation of filaments (U-Net).

YOLO's mask head outputs at stride 4 of a down-scaled image; for 2048 px
observations that caps mask IoU. A U-Net trained at 1024 with a full-resolution
decoder predicts the union of all expert annotations directly, and instances
are recovered afterwards by connected components / watershed.
"""
import argparse, json, random
from pathlib import Path

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

BASE = Path('/data/research/sota_method_2026-10-24/solar_filament')


class SegDataset(Dataset):
    def __init__(self, split, size=1024, files=None, augment=True):
        self.dir = BASE / 'yolo'
        self.split = split
        self.size = size
        self.augment = augment
        self.items = files or sorted(p.stem for p in (self.dir / 'labels' / split).glob('*.txt'))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem = self.items[i]
        img = cv2.imread(str(self.dir / 'images' / self.split / f'{stem}.jpeg'), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mask = np.zeros((h, w), np.uint8)
        for line in (self.dir / 'labels' / self.split / f'{stem}.txt').read_text().splitlines():
            v = np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]
            cv2.fillPoly(mask, [np.round(v).astype(np.int32)], 1)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1]; mask = mask[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1]; mask = mask[::-1]
            k = random.randrange(4)
            if k:
                img = np.rot90(img, k); mask = np.rot90(mask, k)
            if random.random() < 0.3:
                a = random.uniform(0.8, 1.2); b = random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img); mask = np.ascontiguousarray(mask)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--name', default='unet1024')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    device = 'cuda'
    tr = SegDataset('train', a.size, augment=True)
    va = SegDataset('val', a.size, augment=False)
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    dv = DataLoader(va, batch_size=1, shuffle=False, num_workers=2)
    model = smp.Unet(encoder_name=a.encoder, encoder_weights='imagenet', in_channels=3, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    outdir = BASE / 'runs' / a.name
    outdir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        sched.step()
        model.eval(); inter = union = 0.0; dices = []
        with torch.no_grad():
            for x, y in dv:
                x, y = x.to(device), y.to(device)
                p = (torch.sigmoid(model(x)) > 0.5).float()
                i = float((p * y).sum()); u = float(p.sum() + y.sum() - (p * y).sum())
                inter += i; union += u
                dices.append(2 * i / max(float(p.sum() + y.sum()), 1e-6))
        iou = inter / max(union, 1e-6)
        print(f'epoch {ep}/{a.epochs} loss {tot/max(n,1):.4f} val_iou {iou:.4f} dice {np.mean(dices):.4f}', flush=True)
        if iou > best:
            best = iou
            torch.save({'model': model.state_dict(), 'args': vars(a), 'val_iou': iou}, outdir / 'best.pt')
    print('best val_iou', best)


if __name__ == '__main__':
    main()
