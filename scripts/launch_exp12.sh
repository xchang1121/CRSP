#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/bin:/bin"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CR="python3 $ROOT/scripts/colab_retry.py"

python3 -m py_compile "$ROOT/src/cspd_core.py" "$ROOT/src/virtual_tokens_export.py" "$ROOT/src/exp12_official_cspd_and_eos_upgrade.py"
echo "COMPILE_OK"

python3 - "$ROOT/src" "$ROOT/tests" << 'PYEOF'
import io, os, sys, tarfile
src_dir, test_dir = sys.argv[1], sys.argv[2]
with tarfile.open("/tmp/cspd_src.tar.gz", "w:gz") as tf:
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".py"):
            data = open(os.path.join(src_dir, fn), "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(name=fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    for fn in sorted(os.listdir(test_dir)):
        if fn.endswith(".py"):
            data = open(os.path.join(test_dir, fn), "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(name="tests/" + fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
print("Packed /tmp/cspd_src.tar.gz")
PYEOF

cat > /tmp/run_exp12_bg.sh << SHEOF
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_LIB_REPO="${EVAL_LIB_REPO:-}"
mkdir -p /content/cspd
tar -xzf /content/cspd_src.tar.gz -C /content/cspd
if [ -f /content/ref_prompts_manifest.json ]; then
    cp /content/ref_prompts_manifest.json /content/cspd/ref_prompts_manifest.json
fi
if [ -n "\${EVAL_LIB_REPO}" ] && [ ! -d /content/IFBench ]; then
    git clone --depth 1 "\${EVAL_LIB_REPO}" /content/IFBench
fi
python3 - << 'P2'
import torch, importlib, subprocess, sys
needed = ["immutabledict", "langdetect", "nltk", "emoji", "syllapy", "safetensors"]
missing = [p for p in needed if importlib.util.find_spec(p) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
import nltk
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
print("GPU:", torch.cuda.get_device_name(0), "| VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
P2
PYTHONPATH=/content/cspd python3 /content/cspd/tests/test_alignment_and_core.py
exec python3 -u /content/cspd/exp12_official_cspd_and_eos_upgrade.py
SHEOF

if ! colab sessions 2>&1 | grep -q "\[cspd_12\]"; then
    colab new -s cspd_12 --gpu G4 || colab new -s cspd_12 --gpu H100 || colab new -s cspd_12 --gpu A100
fi
$CR upload -s cspd_12 /tmp/cspd_src.tar.gz /content/cspd_src.tar.gz
if [ -n "${REF_PROMPT_MANIFEST:-}" ] && [ -f "${REF_PROMPT_MANIFEST}" ]; then
    $CR upload -s cspd_12 "${REF_PROMPT_MANIFEST}" /content/ref_prompts_manifest.json
fi
$CR upload -s cspd_12 /tmp/run_exp12_bg.sh /content/run_exp12_bg.sh
$CR exec -s cspd_12 -f "$ROOT/scripts/kick_exp12.py"
