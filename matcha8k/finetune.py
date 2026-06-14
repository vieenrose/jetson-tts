"""Fine-tune the Matcha zh-en acoustic model for Taiwan accent on the Qwen3-TTS TW corpus.

Data: (text -> matcha frontend ids, qwen wav -> 16k -> normalized matcha mel). Low-LR full
fine-tune; the corpus is code-mixed so English/code-switch is retained inherently. Saves
fine-tuned checkpoint (re-export to ONNX + re-distill 8k vocoder afterwards).

  python -m matcha8k.finetune --device cuda:0 --out matcha8k/ft_runs/tw --steps 20000
"""
import argparse, os, sys, json, glob, random, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "third_party/Matcha-TTS")
from types import SimpleNamespace
import soundfile as sf, librosa
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.audio import mel_spectrogram
from matcha8k.frontend import MatchaFrontend

MEL = dict(n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000)


def build_model():
    enc = SimpleNamespace(encoder_type="RoPE Encoder",
        encoder_params=SimpleNamespace(n_feats=80, n_channels=192, filter_channels=768,
            filter_channels_dp=256, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
            spk_emb_dim=64, n_spks=1, prenet=True),
        duration_predictor_params=SimpleNamespace(filter_channels_dp=256, kernel_size=3, p_dropout=0.1))
    dec = dict(channels=[256, 256], dropout=0.05, attention_head_dim=64, n_blocks=1,
               num_mid_blocks=2, num_heads=2, act_fn="snakebeta")
    cfm = SimpleNamespace(name="CFM", solver="euler", sigma_min=1e-4)
    # out_size: random mel-segment length for the diffusion loss; MUST be divisible by 4 (U-Net).
    # ~2s at 16k/hop256 = 125 frames -> 128.
    m = MatchaTTS(n_vocab=2190, n_spks=1, spk_emb_dim=64, n_feats=80, encoder=enc, decoder=dec,
                  cfm=cfm, data_statistics={"mel_mean": 0.0, "mel_std": 1.0}, out_size=128,
                  prior_loss=True, use_precomputed_durations=False)
    sd = torch.load("models/matcha-src/pytorch_model.bin", map_location="cpu", weights_only=False)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m


def wav_to_mel(path, mel_mean, mel_std):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != MEL["sampling_rate"]:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=MEL["sampling_rate"], res_type="soxr_hq")
    wav = np.clip(wav, -1.0, 1.0)
    y = torch.from_numpy(wav).unsqueeze(0)
    mel = mel_spectrogram(y, MEL["n_fft"], MEL["num_mels"], MEL["sampling_rate"],
                          MEL["hop_size"], MEL["win_size"], MEL["fmin"], MEL["fmax"], center=False)[0]
    return (mel - mel_mean) / mel_std                       # [80, T] normalized


class TWSet(torch.utils.data.Dataset):
    def __init__(self, root, fe, mel_mean, mel_std, ids_cache):
        self.items = []
        for mf in glob.glob(os.path.join(root, "shard*", "manifest.jsonl")):
            for l in open(mf, encoding="utf-8"):
                try: r = json.loads(l)
                except Exception: continue
                if os.path.exists(r["wav"]):
                    self.items.append((r["id"], r["text"], r["wav"]))
        self.fe, self.mm, self.ms = fe, mel_mean, mel_std
        self.ids_cache = ids_cache

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        uid, text, wav = self.items[i]
        ids = self.ids_cache.get(uid)
        if ids is None:
            # frontend (espeak) in workers can hang; precompute ids offline instead.
            if self.fe is None:
                return torch.zeros(2, dtype=torch.long), torch.zeros(80, 8)
            ids = self.fe.text_to_ids(text)
        x = torch.tensor(ids, dtype=torch.long)
        try:
            y = wav_to_mel(wav, self.mm, self.ms)
        except Exception:
            y = torch.zeros(80, 8)
        return x, y


def collate(batch):
    batch = [(x, y) for x, y in batch if y.shape[1] >= 4 and x.numel() >= 2]
    xs = [b[0] for b in batch]; ys = [b[1] for b in batch]
    xl = torch.tensor([len(x) for x in xs]); yl = torch.tensor([y.shape[1] for y in ys])
    X = torch.zeros(len(xs), max(xl), dtype=torch.long)
    Y = torch.zeros(len(ys), 80, max(yl))
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = x; Y[i, :, :y.shape[1]] = y
    return X, xl, Y, yl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)        # low LR: shift accent, retain English
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="freeze encoder + duration predictor + embeddings, train decoder only (zh-TW accent recipe, arXiv 2305.11320)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    model = build_model().to(dev).train()
    mm, ms = float(model.mel_mean), float(model.mel_std)
    print(f"loaded matcha ckpt; mel_mean {mm:.3f} mel_std {ms:.3f}")
    if args.freeze_encoder:
        # research recipe: freeze content/timing (encoder, duration predictor, embeddings) -> preserves
        # pronunciation/English; train the CFM decoder -> learns the TW-accent acoustic on the qwen manifold
        # (so it stays renderable by the natural-mel vocoder).
        for name, p in model.named_parameters():
            if not name.startswith("decoder."):
                p.requires_grad = False
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"FROZEN encoder/dp/emb; training decoder only: {trn/1e6:.2f}M/{tot/1e6:.1f}M ({100*trn/tot:.1f}%)")
    # load precomputed frontend ids if present (avoids espeak in dataloader workers -> no hang)
    ids_path = os.path.join(args.root + "_ids.json")
    if os.path.exists(ids_path):
        ids_cache = json.load(open(ids_path)); fe = None
        print(f"loaded {len(ids_cache)} precomputed ids; espeak disabled in workers")
    else:
        ids_cache = {}; fe = MatchaFrontend()
    ds = TWSet(args.root, fe, mm, ms, ids_cache)
    print(f"dataset: {len(ds)} clips")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.98))
    step = 0; it = iter(dl); import time; t0 = time.time()
    while step < args.steps:
        try: X, xl, Y, yl = next(it)
        except StopIteration: it = iter(dl); X, xl, Y, yl = next(it)
        X, xl, Y, yl = X.to(dev), xl.to(dev), Y.to(dev), yl.to(dev)
        dur, prior, diff, _ = model(x=X, x_lengths=xl, y=Y, y_lengths=yl, spks=None, out_size=128)
        loss = dur + prior + diff
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step {step} loss {loss.item():.3f} (dur {dur.item():.3f} prior {prior.item():.3f} "
                  f"diff {diff.item():.3f}) {sps:.1f} it/s", flush=True)
        if step % args.save_every == 0:
            out = {"model." + k: v for k, v in model.state_dict().items()}
            torch.save(out, os.path.join(args.out, f"ft_step{step}.bin"))
            torch.save(out, os.path.join(args.out, "last.bin"))
            print(f"saved ckpt @ step {step}", flush=True)
    print("fine-tune done")


if __name__ == "__main__":
    main()
