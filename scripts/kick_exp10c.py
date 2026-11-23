import os
import signal
import subprocess
import time

if os.path.exists("/content/cspd/exp10c.pid"):
    old = open("/content/cspd/exp10c.pid").read().strip()
    if old.isdigit():
        try:
            os.killpg(int(old), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp10c.log", "w") as f:
    f.write("launching exp10c...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp10c_bg.sh"],
    stdout=open("/content/cspd/exp10c.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp10c.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp10c PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
