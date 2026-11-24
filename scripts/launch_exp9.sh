#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/bin:/bin"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

cat > /tmp/run_exp9_bg.sh << 'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /content/cspd
tar -xzf /content/cspd_src.tar.gz -C /content/cspd
echo "=== Setup IFBench & deps ==="
if [ ! -d /content/IFBench ]; then
    if [ -n "${EVAL_LIB_REPO:-}" ]; then git clone --depth 1 "${EVAL_LIB_REPO}" /content/IFBench; fi
fi
python3 -m pip install -q langdetect immutabledict emoji syllapy
python3 - << 'P2'
import nltk, torch
print("GPU:", torch.cuda.get_device_name(0), "| VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
for p in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]:
    try:
        nltk.download(p, quiet=True)
    except Exception:
        pass
P2
echo "=== Launching exp9_multiseed_cr_grpo_and_composition.py ==="
exec python3 -u /content/cspd/exp9_multiseed_cr_grpo_and_composition.py
SHEOF

colab upload -s cspd /tmp/cspd_src.tar.gz /content/cspd_src.tar.gz
colab upload -s cspd /tmp/run_exp9_bg.sh /content/run_exp9_bg.sh

timeout 45 colab exec -s cspd -f "$ROOT/scripts/kick_exp9.py" || (timeout 20 colab restart-kernel -s cspd && sleep 8 && timeout 45 colab exec -s cspd -f "$ROOT/scripts/kick_exp9.py")
