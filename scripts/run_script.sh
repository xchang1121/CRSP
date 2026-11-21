#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_PY="${1:-test_kv_smoke.py}"
LOG_NAME="${2:-kvsmoke}"

bash /mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation/scripts/sync_src.sh
sleep 2

cat > /tmp/launch_generic.py << EOF
import subprocess, os
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/${LOG_NAME}.log", "w")
proc = subprocess.Popen(
    ["python3", "-u", "/content/cspd/${SCRIPT_PY}"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
with open("/content/cspd/${LOG_NAME}.pid", "w") as f:
    f.write(str(proc.pid))
print("Launched ${SCRIPT_PY} PID:", proc.pid)
EOF

timeout 25 colab exec -s cspd -f /tmp/launch_generic.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/launch_generic.py)
