import sys, re, json, glob, os, numpy as np, soundfile as sf, librosa
import sherpa_onnx
M="models/x-asr/deployment/models/chunk-1920ms-model"
rec=sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{M}/encoder-1920ms.onnx",decoder=f"{M}/decoder-1920ms.onnx",joiner=f"{M}/joiner-1920ms.onnx",
    tokens=f"{M}/tokens.txt",num_threads=4,provider="cpu",decoding_method="greedy_search")
def asr(w):
    a,sr=sf.read(w,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=16000: a=librosa.resample(a,orig_sr=sr,target_sr=16000,res_type="soxr_hq")
    s=rec.create_stream(); s.accept_waveform(16000,a)
    tail=np.zeros(int(0.5*16000),dtype=np.float32); s.accept_waveform(16000,tail)
    s.input_finished()
    while rec.is_ready(s): rec.decode_stream(s)
    return rec.get_result(s).strip()
def han(s): return "".join(c for c in s if "一"<=c<="鿿")
def enw(s): return re.findall(r"[A-Za-z][A-Za-z']*",s)
lines={f"{i:02d}_{l.split('|',1)[0]}":l.split('|',1)[1].strip() for i,l in enumerate((x.strip() for x in open('/tmp/accent_lines.txt') if x.strip()),1)}
import collections
agg=collections.defaultdict(lambda:{"en_hit":0,"en_tot":0})
for w in sorted(glob.glob("matcha_eval/accent_ab/*.wav")):
    b=os.path.basename(w).replace(".wav",""); p=b.split("_"); key=f"{p[0]}_{p[1]}"; tag=p[2]
    ref=lines.get(key,"");  hyp=asr(w)
    rw=[x.lower() for x in enw(ref)]; hw=set(x.lower() for x in enw(hyp))
    hit=sum(x in hw for x in rw)
    agg[tag]["en_hit"]+=hit; agg[tag]["en_tot"]+=len(rw)
    print(json.dumps({"id":b,"ref":ref,"hyp":hyp,"en":f"{hit}/{len(rw)}"},ensure_ascii=False))
print("=== English recall by system ===")
for tag,d in agg.items(): print(f"  {tag}: {d['en_hit']}/{d['en_tot']} = {d['en_hit']/max(d['en_tot'],1):.2f}")
