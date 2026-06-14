"""Dataset for the 8 kHz Matcha vocoder: (mel[80,T] @62.5Hz, 8 kHz target wav) pairs.

Training items are random mel-segments with the aligned 8 kHz target slice (128 samples/frame).
"""
import os, json, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

TARGET_SR = 8000
HOP8 = 128                                # samples per mel frame at 8 kHz (8000/62.5)


def scan_items(root, min_frames=16):
    items = []
    for mf in sorted(glob.glob(os.path.join(root, "shard*", "manifest.jsonl"))):
        d = os.path.dirname(mf)
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["mel_frames"] >= min_frames:
                items.append((os.path.join(d, r["id"]), r["mel_frames"]))
    return items


def load_full(base):
    mel = np.load(base + ".mel.npy").astype(np.float32)        # [80,T]
    wav, sr = sf.read(base + ".wav", dtype="float32")
    assert sr == TARGET_SR
    return torch.from_numpy(mel), wav


class MelPairDataset(Dataset):
    def __init__(self, root="matcha_eval/pairs", seg_frames=48, train=True, items=None):
        self.seg = seg_frames
        self.train = train
        self.items = items if items is not None else scan_items(root)
        if not self.items:
            raise RuntimeError(f"no pairs under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        base, T = self.items[i]
        mel = np.load(base + ".mel.npy").astype(np.float32)
        wav, _ = sf.read(base + ".wav", dtype="float32")
        T = mel.shape[1]
        S = self.seg
        if T <= S:
            pad = np.zeros((mel.shape[0], S), np.float32); pad[:, :T] = mel; mel = pad; m0 = 0
        else:
            m0 = np.random.randint(0, T - S) if self.train else 0
            mel = mel[:, m0:m0 + S]
        gen_len = (S - 1) * HOP8                                # ISTFTHead output length
        s0 = m0 * HOP8
        seg = wav[s0:s0 + gen_len]
        if len(seg) < gen_len:
            seg = np.pad(seg, (0, gen_len - len(seg)))
        return torch.from_numpy(mel), torch.from_numpy(seg).unsqueeze(0)


def make_loaders(root="matcha_eval/pairs", seg_frames=48, batch=48, workers=8, n_val=64):
    items = scan_items(root)
    items_sorted = sorted(items, key=lambda x: x[0])
    val = items_sorted[:: max(1, len(items_sorted) // n_val)][:n_val]
    vset = set(b for b, _ in val)
    train_items = [it for it in items if it[0] not in vset]
    ds = MelPairDataset(root, seg_frames=seg_frames, train=True, items=train_items)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    return ds, dl, val
