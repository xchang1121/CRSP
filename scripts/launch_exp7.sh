#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/singm/.local/bin:/usr/bin:/bin"
ROOT="/mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation"

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

cat > /tmp/run_exp7_bg.sh << 'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /content/cspd
tar -xzf /content/cspd_src.tar.gz -C /content/cspd
if [ ! -d /content/IFBench ]; then
    git clone --depth 1 https://github.com/allenai/IFBench.git /content/IFBench
fi
echo "=== Launching exp7_context_robust_sp.py ==="
exec python3 -u /content/cspd/exp7_context_robust_sp.py
SHEOF

colab upload -s cspd /tmp/cspd_src.tar.gz /content/cspd_src.tar.gz
colab upload -s cspd /tmp/run_exp7_bg.sh /content/run_exp7_bg.sh

timeout 30 colab exec -s cspd -f "$ROOT/scripts/kick_exp7.py" || (timeout 15 colab restart-kernel -s cspd && sleep 2 && timeout 30 colab exec -s cspd -f "$ROOT/scripts/kick_exp7.py")
