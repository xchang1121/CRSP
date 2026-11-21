#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cat > /tmp/kill_exp1.py << 'EOF'
import os, signal, time
p = "/content/cspd/exp1.pid"
if os.path.exists(p):
    pid = int(open(p).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    print("Killed exp1 PID", pid)
else:
    print("No exp1 PID")
EOF

timeout 25 colab exec -s cspd -f /tmp/kill_exp1.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/kill_exp1.py)
sleep 2
bash /mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation/scripts/sync_src.sh
sleep 2

cat > /tmp/launch_diag.py << 'EOF'
import subprocess, os
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/diag.log", "w")
proc = subprocess.Popen(
    ["python3", "-u", "/content/cspd/diag_stageA.py"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
with open("/content/cspd/diag.pid", "w") as f:
    f.write(str(proc.pid))
print("Launched diag_stageA.py PID:", proc.pid)
EOF

timeout 25 colab exec -s cspd -f /tmp/launch_diag.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/launch_diag.py)
