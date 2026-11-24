#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/bin:/bin"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CR="python3 $ROOT/scripts/colab_retry.py"

python3 - "$ROOT/src" << 'PYEOF'
import io, os, sys, tarfile
src_dir = sys.argv[1]
with tarfile.open("/tmp/cspd_src.tar.gz", "w:gz") as tf:
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".py"):
            data = open(os.path.join(src_dir, fn), "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(name=fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
print("Packed /tmp/cspd_src.tar.gz")
PYEOF

cat > /tmp/run_exp10c_bg.sh << 'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /content/cspd
tar -xzf /content/cspd_src.tar.gz -C /content/cspd
if [ ! -d /content/IFBench ]; then
    if [ -n "${EVAL_LIB_REPO:-}" ]; then git clone --depth 1 "${EVAL_LIB_REPO}" /content/IFBench; fi
fi
python3 - << 'P2'
import torch, importlib, subprocess, sys
needed = ["immutabledict", "langdetect", "nltk", "emoji", "syllapy"]
missing = [p for p in needed if importlib.util.find_spec(p) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
import nltk
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
print("GPU:", torch.cuda.get_device_name(0), "| VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
P2
exec python3 -u /content/cspd/exp10c_fast_4096_e2e.py
SHEOF

# Allocate best high-VRAM GPU: G4 (Blackwell 96GB) -> H100 (80GB) -> A100 (40GB)
$CR new -s cspd_10c --gpu G4 || $CR new -s cspd_10c --gpu H100 || $CR new -s cspd_10c --gpu A100
$CR upload -s cspd_10c /tmp/cspd_src.tar.gz /content/cspd_src.tar.gz
$CR upload -s cspd_10c /tmp/run_exp10c_bg.sh /content/run_exp10c_bg.sh
$CR exec -s cspd_10c -f "$ROOT/scripts/kick_exp10c.py"
