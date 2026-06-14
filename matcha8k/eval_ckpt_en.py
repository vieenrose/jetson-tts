import sys,os,subprocess,re,numpy as np,onnx
sys.path.insert(0,"third_party/Matcha-TTS"); sys.path.insert(0,".")
ckpt=sys.argv[1]
# export acoustic to onnx sr8000
subprocess.run([sys.executable,"-m","matcha8k.export_acoustic","--ckpt",ckpt,"--out","/tmp/ev_ac.onnx","--n-steps","3"],
               capture_output=True)
m=onnx.load("/tmp/ev_ac.onnx");meta={p.key:p.value for p in m.metadata_props};meta["sample_rate"]="8000"
while len(m.metadata_props):m.metadata_props.pop()
for k,v in meta.items():p=m.metadata_props.add();p.key=k;p.value=str(v)
os.makedirs("/tmp/ev_dropin",exist_ok=True)
onnx.save(m,"/tmp/ev_dropin/model-steps-3.onnx")
for f in ["tokens.txt","lexicon.txt","date-zh.fst","number-zh.fst","phone-zh.fst","vocos-8khz-univ.onnx"]:
    if not os.path.exists(f"/tmp/ev_dropin/{f}"):
        os.symlink(os.path.abspath(f"export/matcha-zh-tw-en-8k-accent/{f}"),f"/tmp/ev_dropin/{f}")
if not os.path.exists("/tmp/ev_dropin/espeak-ng-data"):
    os.symlink(os.path.abspath("export/matcha-zh-tw-en-8k-accent/espeak-ng-data"),"/tmp/ev_dropin/espeak-ng-data")
print("exported", ckpt)
