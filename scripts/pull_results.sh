#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
EXP_NAME="${1:-exp1}"
OUT_LOCAL="/mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation/results"
mkdir -p "$OUT_LOCAL"

cat > /tmp/dump_res.py << EOF
import base64, os
p = "/content/cspd/${EXP_NAME}_results/results.json"
if os.path.exists(p):
    print("B64_START:" + base64.b64encode(open(p, "rb").read()).decode("ascii") + ":B64_END")
else:
    print("NO_RESULTS_YET")
EOF

RAW=$(timeout 25 colab exec -s cspd -f /tmp/dump_res.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/dump_res.py))
python3 - "$RAW" "$OUT_LOCAL/${EXP_NAME}_results.json" << 'PYEOF'
import base64, re, sys
raw, dst = sys.argv[1], sys.argv[2]
m = re.search(r"B64_START:([A-Za-z0-9+/=]+):B64_END", raw)
if not m:
    print("No results JSON found yet.")
    sys.exit(0)
data = base64.b64decode(m.group(1))
open(dst, "wb").write(data)
print(f"Saved {len(data)} bytes to {dst}")
PYEOF
