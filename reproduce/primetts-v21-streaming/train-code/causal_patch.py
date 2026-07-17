"""Causalize the MB-iSTFT generator's convs in-place (streaming Phase 2).

Same state-dict keys/shapes as the symmetric model -> warm-start loads untouched.
With every conv left-causal, chunked inference with per-layer ring-buffer caches is
mathematically identical to full-utterance inference, so we can FINETUNE full-utterance
(simple, reuses train_latest.py) and get streaming correctness for free at export.

- Conv1d with symmetric pad p (= dilation*(k-1)/2) -> pad 2p on the LEFT only.
- ConvTranspose1d (pads (k-u)//2 both sides)       -> padding 0, drop (k-u) RIGHT samples.
- ReflectionPad1d((1,0)) before subband_conv_post   -> already left-only, untouched.

F.pad/slicing are differentiable; instance-level forward monkeypatch survives DDP
(DDP wraps the module object, broadcasting parameters only).
"""
import torch.nn as nn
import torch.nn.functional as F


def causalize_generator(dec, verbose=True):
    n_conv = n_tr = 0
    for m in dec.modules():
        if isinstance(m, nn.ConvTranspose1d):
            k, u = m.kernel_size[0], m.stride[0]
            trim = k - u
            m.padding = (0,)
            base = m.forward
            def tr_fwd(x, base=base, trim=trim):
                y = base(x)
                return y[..., : y.shape[-1] - trim] if trim > 0 else y
            m.forward = tr_fwd
            n_tr += 1
        elif isinstance(m, nn.Conv1d):
            p = m.padding[0]
            if p > 0:
                lpad = 2 * p
                m.padding = (0,)
                base = m.forward
                def c_fwd(x, base=base, lpad=lpad):
                    return base(F.pad(x, (lpad, 0)))
                m.forward = c_fwd
                n_conv += 1
    if verbose:
        print(f"[causal] {n_conv} Conv1d + {n_tr} ConvTranspose1d made left-causal", flush=True)
    return dec
