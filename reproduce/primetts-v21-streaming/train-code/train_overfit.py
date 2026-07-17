"""STEP 5: single-GPU 50-utt overfit trainer for the (phone,tone,lang) MB-iSTFT-VITS.
Full VITS objective (disc + gen: mel/kl/dur/fm/gen/subband) with MAS. fp32 (overfit smoke;
avoids bf16->numpy MAS crash & NaN risk). Run with CUDA_VISIBLE_DEVICES=1 so device cuda:0 == GPU1.
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, "/home/luigi/MB-iSTFT-VITS")
import utils, commons
from data_utils import JsonlTextAudioLoader, JsonlTextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss, subband_stft_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from pqmf import PQMF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", default="/home/luigi/MB-iSTFT-VITS/configs/zhtw_overfit.json")
    ap.add_argument("-m", default="/home/luigi/mbvits_run/overfit_ckpt")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(a.m, exist_ok=True)
    hps = utils.get_hparams_from_file(a.c)
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 => physical GPU1
    torch.manual_seed(hps.train.seed)

    d, tr = hps.data, hps.train
    hop, seg = d.hop_length, tr.segment_size
    ds = JsonlTextAudioLoader(d.training_files, d)
    dl = DataLoader(ds, batch_size=tr.batch_size, shuffle=True, num_workers=2,
                    collate_fn=JsonlTextAudioCollate(), drop_last=False, pin_memory=True)
    print(f"[train] {len(ds)} utts, batch={tr.batch_size}, steps={a.steps}", flush=True)

    net_g = SynthesizerTrn(88, d.filter_length // 2 + 1, seg // hop, **hps.model.__dict__).cuda(dev)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(dev)
    print(f"[train] net_g params = {sum(p.numel() for p in net_g.parameters())/1e6:.2f}M", flush=True)
    optim_g = torch.optim.AdamW(net_g.parameters(), tr.learning_rate, betas=tr.betas, eps=tr.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), tr.learning_rate, betas=tr.betas, eps=tr.eps)
    net_g.train(); net_d.train()

    step = 0; t0 = time.time()
    while step < a.steps:
        for batch in dl:
            phone, tone, lang, xlen, spec, slen, wav, wlen = [b.cuda(dev, non_blocking=True) for b in batch]

            y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, tone, lang, xlen, spec, slen)

            mel = spec_to_mel_torch(spec, d.filter_length, d.n_mel_channels, d.sampling_rate, d.mel_fmin, d.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, seg // hop)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), d.filter_length, d.n_mel_channels,
                                              d.sampling_rate, hop, d.win_length, d.mel_fmin, d.mel_fmax)
            y = commons.slice_segments(wav, ids_slice * hop, seg)

            # ---- discriminator ----
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            optim_d.zero_grad(); loss_disc.backward()
            commons.clip_grad_value_(net_d.parameters(), None); optim_d.step()

            # ---- generator ----
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * tr.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * tr.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            y_mb = PQMF(y.device).analysis(y)
            loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
            optim_g.zero_grad(); loss_gen_all.backward()
            commons.clip_grad_value_(net_g.parameters(), None); optim_g.step()

            if step % tr.log_interval == 0:
                dt = time.time() - t0
                print(f"step {step:5d}  gen_all={loss_gen_all.item():7.3f}  mel={loss_mel.item():6.3f}  "
                      f"kl={loss_kl.item():6.3f}  dur={loss_dur.item():6.3f}  fm={loss_fm.item():6.3f}  "
                      f"gen={loss_gen.item():5.3f}  sub={loss_subband.item():5.3f}  disc={loss_disc.item():5.3f}  "
                      f"({dt/max(1,step):.2f}s/it)", flush=True)
            step += 1
            if step % 2000 == 0:
                torch.save({"model": net_g.state_dict(), "step": step, "config": a.c},
                           os.path.join(a.m, f"G_{step}.pth"))
                print(f"[train] saved G_{step}.pth", flush=True)
            if step >= a.steps:
                break

    ckpt = os.path.join(a.m, "G_last.pth")
    torch.save({"model": net_g.state_dict(), "step": step, "config": a.c}, ckpt)
    print(f"[train] DONE {step} steps in {(time.time()-t0)/60:.1f} min -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
