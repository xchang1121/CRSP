#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/bin:/bin"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CR="python3 $ROOT/scripts/colab_retry.py"

python3 -m py_compile "$ROOT/src/exp10c_fast_4096_e2e.py"
echo "COMPILE_OK"

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
print("Repacked /tmp/cspd_src.tar.gz")
PYEOF

$CR upload -s cspd_10c /tmp/cspd_src.tar.gz /content/cspd_src.tar.gz
$CR exec -s cspd_10c -f "$ROOT/scripts/kick_exp10c.py"
