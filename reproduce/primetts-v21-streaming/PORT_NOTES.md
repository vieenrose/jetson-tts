# MB-iSTFT-VITS (zh-TW/en) — ggml-CUDA port notes (dev-box prep)

Target: Jetson Nano gen1 GPU, **sm_53, no CUDA Graphs**, ggml-CUDA (RapidSpeech.cpp
already ports conv_transpose→zero-stuff+conv1d and the iSTFT). Model ≈ 34.7M total;
**inference subset = 27.45M params / 283 learned tensors** (+3 baked DSP tensors).

Source of truth on the dev box:
- Model code: `/home/luigi/MB-iSTFT-VITS/models.py`, `attentions.py`, `modules.py`, `pqmf.py`, `stft.py`
- Config: `/home/luigi/MB-iSTFT-VITS/configs/zhtw_mb_istft_16k.json`
- Checkpoints: `/home/luigi/MB-iSTFT-VITS/logs/zhtw_mbistft_16k/G_*.pth` (tools auto-pick latest)
- Frontend: `/home/luigi/jetson-tts/mossnano/zhtw8k/frontend_bopomofo.py` (88 symbols, 6 tones, 2 langs)

Tooling produced (all tested):
- `tools/convert_mbistft_to_gguf.py` → `/home/luigi/mbvits_run/mbistft_zhtw_16k.gguf` (+`.manifest.json`)
- `tools/gen_parity_inputs.py`      → `/home/luigi/mbvits_run/parity_inputs.json` (run in `.venv-breezy`, CPU)
- `tools/dump_parity_refs.py`       → `/home/luigi/mbvits_run/parity_refs/*.npy` (+`parity_manifest.json`) (run `.venv`, GPU1)

---

## 0. Architecture hyper-params (config)
```
n_vocab=88  num_tones=6  num_langs=2         (emb_phone / emb_tone / emb_lang)
hidden_channels=192  inter_channels=192  filter_channels(FFN)=768
n_heads=2  n_layers=6  kernel_size(FFN)=3    k_channels = 192/2 = 96
window_size=4  (rel-pos attention; NOT in json, it is the attentions.Encoder default)
DurationPredictor: filter_channels=256, kernel=3   (deterministic; use_sdp=false)
flow: n_flows=4  WN n_layers=4  kernel=5  dilation_rate=1 (→ all dilations=1)  mean_only=true
decoder (Multiband_iSTFT_Generator):
  upsample_rates=[4,4]  upsample_kernel_sizes=[16,16]  upsample_initial_channel=512
  resblock="1"  resblock_kernel_sizes=[3,7,11]  resblock_dilation_sizes=[[1,3,5]]*3
  gen_istft_n_fft=16  gen_istft_hop_size=4  subbands=4
sampling_rate=16000  hop_length=256  ( = 4*4 * 4(subbands) * 4(istft_hop) )  filter_length=1024
add_blank=true → inputs are blank-interleaved (id 0) on ALL THREE streams (see §5)
```

---

## 1. INFERENCE-GRAPH INVENTORY (SynthesizerTrn.infer)

Run at inference (KEEP): `enc_p` (TextEncoder), `dp` (deterministic DurationPredictor),
`flow` (ResidualCouplingBlocks, **run in REVERSE**), `dec` (Multiband-iSTFT generator).

**EXCLUDED — training-only (do NOT export):**
- `enc_q` PosteriorEncoder (pre + 16-layer WN + proj) — 100 tensors — posterior over spectrogram, never used at infer.
- `MultiPeriodDiscriminator` (DiscriminatorS + 5×DiscriminatorP) — lives in `D_*.pth`, not in `G_*.pth`.
- `StochasticDurationPredictor` — absent (use_sdp=false; `dp` is deterministic, 10 tensors, NO `.flows.`).
- optimizer / iteration / lr in the ckpt dict.

### 1a. enc_p — TextEncoder  (113 tensors; NO weight_norm)
| tensor | shape | note |
|---|---|---|
| `enc_p.emb_phone.weight` | (88,192) | phone embedding (get_rows) |
| `enc_p.emb_tone.weight` | (6,192) | tone embedding |
| `enc_p.emb_lang.weight` | (2,192) | lang embedding |
| per layer i=0..5 `attn_layers.i.conv_{q,k,v,o}.{weight,bias}` | w (192,192,1) b (192,) | 1×1 conv = linear |
| per layer i `attn_layers.i.emb_rel_{k,v}` | (1,9,96) | Shaw rel-pos emb, heads-shared |
| per layer i `norm_layers_1.i.{gamma,beta}` | (192,) | LayerNorm (post-attn) |
| per layer i `ffn_layers.i.conv_1.{weight,bias}` | w (768,192,3) | FFN in→filter, k=3 same-pad |
| per layer i `ffn_layers.i.conv_2.{weight,bias}` | w (192,768,3) | FFN filter→out, k=3 same-pad |
| per layer i `norm_layers_2.i.{gamma,beta}` | (192,) | LayerNorm (post-FFN) |
| `enc_p.proj.{weight,bias}` | w (384,192,1) | → split into m_p, logs_p (192 each) |

Forward: `x = (emb_phone(id)+emb_tone(tone)+emb_lang(lang)) * sqrt(192)` → transpose to [1,192,t]
→ 6× `{ x = LN1(x + MHA(x)); x = LN2(x + FFN(x)) }` → `proj` → split → `m_p, logs_p`.
(dropout is eval-time no-op.)

### 1b. dp — DurationPredictor  (10 tensors; deterministic; NO weight_norm, NO cond since gin=0)
`conv_1 (256,192,3) pad1 → relu → LN(256) → conv_2 (256,256,3) pad1 → relu → LN(256) → proj (1,256,1)`.
`logw = dp(enc, x_mask)`; `w = exp(logw)*x_mask`; `w_ceil = ceil(w)` → integer per-phone frame counts.

### 1c. flow — ResidualCouplingBlock  (80 tensors after fold; REVERSE order at infer)
`flow.flows` index [0,2,4,6] = ResidualCouplingLayer, [1,3,5,7] = Flip (no params).
At infer: iterate flows **reversed**; each Flip = `torch.flip(x, dim=1)`; each coupling layer (mean_only, reverse):
```
x0,x1 = split(x, 96/96, dim=1)
h  = pre(x0)                         # Conv1d (192,96,1)  plain
h  = WN(h, x_mask, g=None)           # 4 layers, see below
m  = post(h)                         # Conv1d (96,192,1)  plain ; logs=0 (mean_only)
x1 = (x1 - m) * exp(-0) = x1 - m     # reverse, logs=0
x  = cat(x0, x1, dim=1)
```
WN layer j=0..3 (`in_layers.j` (384,192,5) pad2 dilation=1; `res_skip_layers.j` — (384,192,1) for j<3, (192,192,1) for j=3):
```
xin = in_layers[j](x)                      # conv1d k=5 pad2
acts = tanh(xin[:,:192]) * sigmoid(xin[:,192:])   # fused (g=0, no cond)
rs = res_skip_layers[j](acts)
if j<3: x = (x + rs[:,:192]) * x_mask ; output += rs[:,192:]
else:   output += rs
return output * x_mask
```
NOTE: dilation_rate=1 ⇒ every WN conv dilation = 1 (no dilated conv in the flow).

### 1d. dec — Multiband_iSTFT_Generator  (80 tensors after fold; + baked DSP)
```
x = conv_pre(z)                              # Conv1d (512,192,7) pad3
for i in 0,1:
  x = leaky_relu(x, 0.1); x = ups[i](x)      # ConvTranspose1d: ups0 (512,256,16) s4 pad6 ; ups1 (256,128,16) s4 pad6
  x = mean_j resblock[i*3+j](x)              # 3 ResBlock1 per stage (kernels 3,7,11)
x = leaky_relu(x, 0.1)
x = reflection_pad1d((1,0))(x)               # left reflect by 1  ← see §4/net-new
x = subband_conv_post(x)                     # Conv1d (72,128,7) pad3   (72 = 4*(16+2))
x = reshape(x, [1, 4, 18, T])                # subbands=4, 18 = n_fft+2
spec  = exp(x[:, :, :9,  :])                 # 9 = n_fft//2+1
phase = pi * sin(x[:, :, 9:, :])             # 9 channels
y_mb  = iSTFT(spec, phase)                    # per (B*4) group; n_fft=16 hop=4 win=hann(16) → [1,4,1,Ls], squeeze→[1,4,Ls]
wav   = PQMF.synthesis(y_mb)                  # → [1,1,4*Ls]
```
ResBlock1 (ch=256 for i=0 / ch=128 for i=1), each of kernels k∈{3,7,11}:
```
for (c1,c2) in zip(convs1[dil=1,3,5], convs2[dil=1,1,1]):
    xt = c1(leaky_relu(x,0.1)); xt = c2(leaky_relu(xt,0.1)); x = x + xt
```
convs1 dilations = (1,3,5) **(dilated conv1d needed)**; convs2 dilations = 1. All pad = get_padding(k,dil).

---

## 2. RELATIVE-POSITION MULTI-HEAD ATTENTION — exact spec (THE riskiest kernel)

Self-attention, `n_heads=2`, `k_channels=96`, `window_size=4`, `heads_share=True`
(so `emb_rel_k/v` first dim = 1, broadcast over both heads). Shapes below for one utterance,
sequence length `t` (= blank-interleaved phone count; ≤ ~90 in the parity set).

Projections are 1×1 convs (= linear): `q=conv_q(x); k=conv_k(x); v=conv_v(x)`, each [1,192,t].
Reshape to heads: `[1,192,t] → [1, 2, 96, t] → transpose(2,3) → [1, 2, t, 96]`.

**Content scores** (standard): `scores = (q / sqrt(96)) @ kᵀ`  → [1,2,t,t].

**Relative-position K term** (added to scores):
1. `rel_k = _get_relative_embeddings(emb_rel_k, t)` — pad+slice the (1,9,96) table to (1, 2t−1, 96):
   ```
   max_rel = 2*window_size+1 = 9
   pad = max(t-(window_size+1), 0) = max(t-5,0)          # pad the position axis both sides
   start = max((window_size+1)-t, 0) = max(5-t,0)
   emb' = F.pad(emb_rel_k, [[0,0],[pad,pad],[0,0]]) if pad>0 else emb_rel_k
   rel_k = emb'[:, start : start + 2t-1, :]              # → (1, 2t-1, 96)
   ```
2. `rel_logits = (q/sqrt(96)) @ rel_kᵀ`  → [1,2,t, 2t−1]   (`_matmul_with_relative_keys`)
3. `scores_local = _relative_position_to_absolute_position(rel_logits)` → [1,2,t,t]  **(skew — net-new, see below)**
4. `scores += scores_local`

Mask: `scores = masked_fill(mask==0, -1e4)` (mask = x_mask outer-product; for single utt all ones ⇒ no-op).
`p_attn = softmax(scores, dim=-1)`  → [1,2,t,t].

**Output**: `out = p_attn @ v`  → [1,2,t,96].

**Relative-position V term** (added to out):
1. `rel_weights = _absolute_position_to_relative_position(p_attn)` → [1,2,t, 2t−1]  **(skew — net-new)**
2. `rel_v = _get_relative_embeddings(emb_rel_v, t)` → (1, 2t−1, 96)
3. `out += rel_weights @ rel_v`  → [1,2,t,96]   (`_matmul_with_relative_values`)

Merge heads: `out.transpose(2,3).contiguous().view(1,192,t)` → `conv_o` → [1,192,t].

### The two SKEW ops (pad → reshape → slice; exact, from attentions.py)
`_relative_position_to_absolute_position(x)`  `x:[b,h,t,2t-1] → [b,h,t,t]`:
```
x = F.pad(x, [0,1])                       # pad LAST dim RIGHT by 1   → [b,h,t,2t]
x = x.view(b,h, t*2t)                     # flatten
x = F.pad(x, [0, t-1])                    # pad flat RIGHT by t-1
x = x.view(b,h, t+1, 2t-1)[:, :, :t, t-1:]   # reshape + slice → [b,h,t,t]
```
`_absolute_position_to_relative_position(x)`  `x:[b,h,t,t] → [b,h,t,2t-1]`:
```
x = F.pad(x, [0, t-1])                    # pad LAST dim RIGHT by t-1 → [b,h,t,2t-1]
x = x.view(b,h, t*t + t*(t-1))            # flatten
x = F.pad(x, [t, 0])                      # pad flat LEFT by t         ← LEFT pad (ggml_pad can't; see §4)
x = x.view(b,h, t, 2t)[:, :, :, 1:]       # reshape + slice → [b,h,t,2t-1]
```
Both are pure index gymnastics (pad+view+slice); all matmuls/softmax are reused ggml ops.
**Validate the ggml implementation against `parity_refs/uttNN_attn0..5.npy`** (cosine > 0.99), attn0 first.

---

## 3. PQMF synthesis + iSTFT (fixed DSP — baked into the GGUF, so no runtime re-derivation)

The Kaiser prototype and cosine-modulated filters are NOT learned; they are recomputed on the
dev box and **baked into the GGUF** so the C++ side just loads plain conv weights:
- `pqmf.synthesis_filter` (4,1,63) — for `conv1d`
- `pqmf.updown_filter`    (4,4,4)  — for `conv_transpose1d`
- `istft.window`          (16,)    — hann, fftbins=True, for the iSTFT

`PQMF.synthesis(y_mb[1,4,Ls])` (pqmf.py) maps to exactly two ops already in the kernel set:
```
x   = conv_transpose1d(y_mb, updown_filter * subbands(=4), stride=4)   # [1,4,4*Ls]
wav = conv1d( pad(x, taps//2 = 31 zeros both sides), synthesis_filter ) # [1,1,4*Ls]
```
Prototype design params (fixed): `taps=62, cutoff_ratio=0.15, beta=9.0, subbands=4`.

iSTFT = `torch.istft(spec·e^{i·phase}, n_fft=16, hop=4, win_length=16, window=hann, center=True, onesided=True, normalized=False)`.
**Match torch.istft semantics exactly**: center padding (n_fft//2=8 reflect at edges then trim) and the
COLA window-sumsquare normalization (divide overlap-add by Σ win²). RapidSpeech's iSTFT must reproduce this;
validate against `parity_refs/uttNN_o_mb.npy` (per-subband, pre-PQMF) then `..._wav.npy`.

---

## 4. NET-NEW vs REUSED ggml-CUDA ops

**REUSED (already present in ggml-cuda / RapidSpeech.cpp):**
`ggml_get_rows` (3 embeddings) · `ggml_mul_mat` (1×1 convs, attention, expand-matmuls) ·
`ggml_conv_1d` incl **dilation** (FFN k=3, WN k=5, resblock dil 1/3/5, conv_pre/subband_post k=7) ·
`ggml_conv_transpose_1d` (dec ups k=16 s4, PQMF updown) · **iSTFT** (n_fft=16) ·
`ggml_norm`+scale/shift (LayerNorm over 192 ch) · `ggml_soft_max` · `ggml_add/mul/scale` ·
`ggml_transpose/reshape/view/cont` · `ggml_tanh/sigmoid/exp` · `ggml_leaky_relu(0.1)` ·
`ggml_sin` (phase) · `ggml_pad` (zero, END-padding).
NOTE: **no gelu / no dilated-WN / no RNG needed** — DDSConv/ConvFlow (gelu) live only in the SDP/ConvFlow
which are NOT in the inference path, and noise_scale=0 removes all sampling.

**NET-NEW (implement + validate for this arch):**
1. **Rel-pos skew reindex** (`rel↔abs`, §2). The matmuls/softmax are reused; the *skew* is new.
   Blocker detail: `_absolute_position_to_relative_position` needs a **LEFT pad** on the flat axis
   (`F.pad(...,[t,0])`), which `ggml_pad` (end-only) can't do directly — emulate with a concat of a
   zero block in front, or pad-then-roll, then `view`+`slice`. This is the single genuinely-new kernel/graph
   fragment; get it right first (check attn0 ref).
2. **Reflection pad (1,0)** before `subband_conv_post` — ggml only zero-pads. It is a 1-sample left
   reflection; emulate by prepending a copy of column 1 (or a 3-line custom op). Minor but silent if skipped.
3. **iSTFT COLA/window normalization + center trimming** must match `torch.istft` bit-for-bit (§3). If the
   existing RapidSpeech iSTFT was tuned for a different n_fft/window it may mismatch — validate on `o_mb`.
4. **Length regulator / `generate_path`** (`cumsum(w_ceil)` → `sequence_mask` → diff → path, plus `ceil`):
   recommend doing on the **host/CPU** ([1,1,t] is tiny) to avoid a GPU `ceil` + gather/scatter path.
   The expand step `m_p_exp = attn @ m_pᵀ` can stay on GPU (reused mul_mat) or be a host gather.

---

## 5. Input contract (must match training)
Training used `add_blank=true`: the 3 id streams are **blank-interleaved with id 0**
(`intersperse(seq,0)` on phone AND tone AND lang) before the model. `parity_inputs.json` already
stores the interleaved streams (`phone_ids/tone_ids/lang_ids`) plus the raw ones. The ggml port must
apply the same interleave to frontend output. blank_id=0. Tones 0..5, langs {ZH:0, EN:1}.

---

## 6. sm_53 flags
- **Keep F32 end-to-end.** Do NOT rely on fp16 mul_mat paths (sm_53 fp16 is weak / poorly covered in
  ggml-cuda). All parity refs are F32; GGUF is F32.
- **No CUDA Graphs** (given for Nano gen1) — build a static graph, launch per-op.
- Tensors are tiny (t ≤ ~90 text frames, ≤ ~314 acoustic frames, ≤ 512 ch). No tiling/occupancy concern;
  **the risk is correctness (skew + iSTFT), not throughput.** Largest activation ≈ 512×~1900 f32 (~4 MB).
- Confirm the conv kernel honors **dilation** (resblock 3/5) and **stride-4 conv_transpose k=16** on sm_53.

---

## 7. How to (re)generate at final convergence
```
# 1) inputs (CPU; needs g2pw+g2p_en → .venv-breezy)
CUDA_VISIBLE_DEVICES="" NLTK_DATA=/home/luigi/nltk_data \
  /home/luigi/jetson-tts/.venv-breezy/bin/python tools/gen_parity_inputs.py
# 2) gguf (CPU; .venv) — auto-picks latest G_*.pth
/home/luigi/jetson-tts/.venv/bin/python tools/convert_mbistft_to_gguf.py \
  --out /home/luigi/mbvits_run/mbistft_zhtw_16k.gguf
# 3) parity refs (GPU1; .venv) — PQMF hardcodes .cuda
CUDA_VISIBLE_DEVICES=1 /home/luigi/jetson-tts/.venv/bin/python tools/dump_parity_refs.py
```
GGUF tensor names == PyTorch state_dict keys (weight_g/weight_v folded to `.weight`); the C++ loader
indexes by these names. Verified: fold is bit-exact vs the weight_norm modules (max err 0.0), and the
GGUF holds EXACTLY the 283 inference params (no enc_q, no missing/extra) + 3 baked DSP tensors.
