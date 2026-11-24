import os
import signal
import subprocess
import time

if os.path.exists("/content/cspd/exp12.pid"):
    old = open("/content/cspd/exp12.pid").read().strip()
    if old.isdigit():
        try:
            os.killpg(int(old), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp12.log", "w") as f:
    f.write("launching exp12...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp12_bg.sh"],
    stdout=open("/content/cspd/exp12.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp12.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp12 PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
