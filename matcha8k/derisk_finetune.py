"""De-risk: can we load dengcunqin's matcha zh-en pytorch_model.bin into MatchaTTS and train it?
Instantiates MatchaTTS (vocab 2190, n_spks=1), loads the checkpoint, runs forward + backward.
"""
import sys, torch
sys.path.insert(0, "third_party/Matcha-TTS")
from types import SimpleNamespace
from matcha.models.matcha_tts import MatchaTTS

# config from configs/model/*.yaml, vocab from the checkpoint emb (2190)
encoder = SimpleNamespace(
    encoder_type="RoPE Encoder",
    encoder_params=SimpleNamespace(n_feats=80, n_channels=192, filter_channels=768,
        filter_channels_dp=256, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
        spk_emb_dim=64, n_spks=1, prenet=True),
    duration_predictor_params=SimpleNamespace(filter_channels_dp=256, kernel_size=3, p_dropout=0.1),
)
decoder = dict(channels=[256, 256], dropout=0.05, attention_head_dim=64,
    n_blocks=1, num_mid_blocks=2, num_heads=2, act_fn="snakebeta")
cfm = SimpleNamespace(name="CFM", solver="euler", sigma_min=1e-4)
data_stats = {"mel_mean": 0.0, "mel_std": 1.0}  # overwritten by checkpoint's baked-in stats

model = MatchaTTS(n_vocab=2190, n_spks=1, spk_emb_dim=64, n_feats=80,
    encoder=encoder, decoder=decoder, cfm=cfm, data_statistics=data_stats,
    out_size=None, prior_loss=True, use_precomputed_durations=False)
n = sum(p.numel() for p in model.parameters())
print(f"MatchaTTS instantiated: {n/1e6:.2f}M params")

sd = torch.load("models/matcha-src/pytorch_model.bin", map_location="cpu", weights_only=False)
sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing {len(missing)}, unexpected {len(unexpected)}")
if missing[:5]: print("  missing e.g.:", missing[:5])
if unexpected[:5]: print("  unexpected e.g.:", unexpected[:5])
print(f"mel_mean={float(model.mel_mean):.4f} mel_std={float(model.mel_std):.4f}")

# forward + backward on a dummy batch (the trainability test)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(dev).train()
B, L, T = 2, 30, 160
x = torch.randint(1, 2190, (B, L), device=dev)
x_lengths = torch.tensor([L, L - 4], device=dev)
y = torch.randn(B, 80, T, device=dev)
y_lengths = torch.tensor([T, T - 20], device=dev)
dur_loss, prior_loss, diff_loss, attn = model(x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, spks=None)
loss = dur_loss + prior_loss + diff_loss
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
print(f"FORWARD+BACKWARD OK on {dev}: dur={float(dur_loss):.3f} prior={float(prior_loss):.3f} diff={float(diff_loss):.3f}")
print(f"grad norm {float(gnorm):.2f} -> checkpoint is TRAINABLE. De-risk PASS.")
