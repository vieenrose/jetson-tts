"""v2.2 multi-speaker streaming split export.
EncWrap: (x,tone,lang,x_lengths,noise_scale,length_scale,sid) -> (z, g)   [run once]
DecWrap: (z_chunk, g) -> wav                                              [per chunk, overlap-save]
Clean non-causal vocoder (NO causalize); stream with chunk=24/left=64/RIGHT=16.
The speaker embedding g is constant per utterance: enc emits it, dec consumes it each chunk.
"""
import argparse, json, os, sys
import torch
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,_ROOT)
from models import SynthesizerTrn
from tools.export_onnx_stream_split import install_onnx_dsp

class EncWrap(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net=net
    def forward(self, x, tone, lang, x_lengths, noise_scale, length_scale, sid):
        o,o_mb,attn,y_mask,(z,z_p,m_p,logs_p)=self.net.infer(
            x,tone,lang,x_lengths,sid=sid,noise_scale=noise_scale,length_scale=length_scale)
        return z*y_mask   # dec is speaker-agnostic (g only conditions flow/enc)

class DecWrap(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net=net
    def forward(self, z):
        return self.net.dec(z)[0]             # [1,1,F*256] (dec ignores g)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/luigi/mbvits_run/keep_v22_35500_G.pth")
    ap.add_argument("--config", default=os.path.join(_ROOT,"configs","zhtw_mb_istft_16k_v22.json"))
    ap.add_argument("--outdir", default="/home/luigi/mbvits_run/v22_split")
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    cfg=json.load(open(a.config)); m,d=cfg["model"],cfg["data"]
    net=SynthesizerTrn(88,d["filter_length"]//2+1,cfg["train"]["segment_size"]//d["hop_length"],**m)
    sd=torch.load(a.ckpt,map_location="cpu",weights_only=False)["model"]
    net.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in sd.items()},strict=True)
    net.eval(); net.dec.remove_weight_norm(); install_onnx_dsp(m)   # NO causalize = clean vocoder

    T=33
    ex=(torch.randint(1,87,(1,T)),torch.randint(0,6,(1,T)),torch.randint(0,2,(1,T)),
        torch.tensor([T]),torch.tensor([0.0]),torch.tensor([1.0]),torch.tensor([0]))
    enc=os.path.join(a.outdir,"v22_enc.onnx")
    torch.onnx.export(EncWrap(net),ex,enc,opset_version=17,dynamo=False,
        input_names=["x","tone","lang","x_lengths","noise_scale","length_scale","sid"],
        output_names=["z"],
        dynamic_axes={"x":{1:"T"},"tone":{1:"T"},"lang":{1:"T"},"z":{2:"F"}})
    with torch.no_grad(): z=EncWrap(net)(*ex)
    print(f"[enc] {enc}  z={tuple(z.shape)}")

    dec=os.path.join(a.outdir,"v22_dec.onnx")
    zex=torch.randn(1,m["inter_channels"],92)
    torch.onnx.export(DecWrap(net),(zex,),dec,opset_version=17,dynamo=False,
        input_names=["z"],output_names=["wav"],dynamic_axes={"z":{2:"F"},"wav":{2:"L"}})
    with torch.no_grad(): w=DecWrap(net)(zex)
    print(f"[dec] {dec}  wav={tuple(w.shape)} ({w.shape[2]//zex.shape[2]}/frame)")

if __name__=="__main__": main()
