"""Distillation dataset: (z[192,T] @86.13Hz, 8kHz target wav) pairs from data/pairs/shard*/.

Training items are random z-segments of fixed length with the aligned 8 kHz target slice.
g is a single constant vector (single speaker) loaded once and shared.
"""
import os, json, glob, numpy as np, torch
from torch.utils.data import Dataset
import soundfile as sf

from .audio_config import Z_FRAME_RATE, TARGET_SR, HOP, inter_frames_for

SAMPLES_PER_ZFRAME = TARGET_SR / Z_FRAME_RATE   # 92.8798...


def scan_items(root, min_frames=8):
    items = []
    for mf in sorted(glob.glob(os.path.join(root, "shard*", "manifest.jsonl"))):
        d = os.path.dirname(mf)
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["z_frames"] >= min_frames:
                items.append((os.path.join(d, r["id"]), r["z_frames"]))
    return items


def load_g(root):
    g = np.load(os.path.join(root, "g.npy")).astype(np.float32)        # [256]
    return torch.from_numpy(g).view(256, 1)


class PairDataset(Dataset):
    def __init__(self, root="data/pairs", seg_z=64, train=True, items=None):
        self.root = root
        self.seg_z = seg_z
        self.train = train
        self.items = items if items is not None else scan_items(root)
        self.g = load_g(root)
        if not self.items:
            raise RuntimeError(f"no pairs under {root}")

    def __len__(self):
        return len(self.items)

    def _load(self, base):
        z = np.load(base + ".z.npy").astype(np.float32)               # [192,T]
        wav, sr = sf.read(base + ".wav", dtype="float32")
        assert sr == TARGET_SR
        return z, wav

    def __getitem__(self, i):
        base, T = self.items[i]
        z, wav = self._load(base)
        T = z.shape[1]
        S = self.seg_z
        if T <= S:
            z0 = 0
            zpad = np.zeros((z.shape[0], S), np.float32)
            zpad[:, :T] = z
            z = zpad
            Tcrop = S
        else:
            z0 = np.random.randint(0, T - S) if self.train else 0
            z = z[:, z0:z0 + S]
            Tcrop = S
        gen_len = inter_frames_for(Tcrop) * HOP
        s0 = int(round(z0 * SAMPLES_PER_ZFRAME))
        seg = wav[s0:s0 + gen_len]
        if len(seg) < gen_len:
            seg = np.pad(seg, (0, gen_len - len(seg)))
        return torch.from_numpy(z), torch.from_numpy(seg).unsqueeze(0)  # [192,S], [1,gen_len]


def load_full(base):
    """Full-utterance (z, wav8k) for validation/PESQ. z:[192,T] wav:[S]."""
    z = np.load(base + ".z.npy").astype(np.float32)
    wav, sr = sf.read(base + ".wav", dtype="float32")
    assert sr == TARGET_SR
    return torch.from_numpy(z), wav


def make_loaders(root="data/pairs", seg_z=64, batch=32, workers=8, n_val=64):
    from torch.utils.data import DataLoader
    items = scan_items(root)
    items_sorted = sorted(items, key=lambda x: x[0])
    val = items_sorted[:: max(1, len(items_sorted) // n_val)][:n_val]
    val_set = set(b for b, _ in val)
    train_items = [it for it in items if it[0] not in val_set]
    ds = PairDataset(root, seg_z=seg_z, train=True, items=train_items)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    return ds, dl, val
