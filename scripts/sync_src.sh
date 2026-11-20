#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="/mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation"

# Pack all .py files in src/ (stripping any CR) into base64 tar.gz and extract to /content/cspd
TMP_PY="/tmp/cspd_sync_upload.py"
python3 - "$ROOT/src" "$TMP_PY" << 'PYEOF'
import base64, io, os, sys, tarfile

src_dir, out_py = sys.argv[1], sys.argv[2]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".py"):
            p = os.path.join(src_dir, fn)
            data = open(p, "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(name=fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
code = f"""import base64, io, os, tarfile
os.makedirs('/content/cspd', exist_ok=True)
buf = io.BytesIO(base64.b64decode({b64!r}))
with tarfile.open(fileobj=buf, mode='r:gz') as tf:
    tf.extractall('/content/cspd')
print('Synced /content/cspd:', sorted(os.listdir('/content/cspd')))
"""
open(out_py, "w").write(code)
PYEOF

colab exec -s cspd -f "$TMP_PY"
