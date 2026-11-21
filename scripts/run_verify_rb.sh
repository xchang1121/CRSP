#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cat > /tmp/kill_diag.py << 'EOF'
import os, signal, time
p = "/content/cspd/diag.pid"
if os.path.exists(p):
    pid = int(open(p).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    print("Killed diag PID", pid)
EOF

timeout 25 colab exec -s cspd -f /tmp/kill_diag.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/kill_diag.py)
sleep 2
bash /mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation/scripts/sync_src.sh
sleep 2

cat > /tmp/launch_vrb.py << 'EOF'
import subprocess, os
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/vrb.log", "w")
proc = subprocess.Popen(
    ["python3", "-u", "/content/cspd/verify_rb_cspd.py"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
with open("/content/cspd/vrb.pid", "w") as f:
    f.write(str(proc.pid))
print("Launched verify_rb_cspd.py PID:", proc.pid)
EOF

timeout 25 colab exec -s cspd -f /tmp/launch_vrb.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 25 colab exec -s cspd -f /tmp/launch_vrb.py)
