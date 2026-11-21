#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

bash /mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation/scripts/sync_src.sh
sleep 2

cat > /tmp/launch_exp1b.py << 'EOF'
import subprocess, os
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/exp1b.log", "w")
proc = subprocess.Popen(
    ["python3", "-u", "/content/cspd/exp1b_cspd_final.py"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
with open("/content/cspd/exp1b.pid", "w") as f:
    f.write(str(proc.pid))
print("Launched exp1b_cspd_final.py PID:", proc.pid)
EOF

timeout 25 colab exec -s cspd -f /tmp/launch_exp1b.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/launch_exp1b.py)
