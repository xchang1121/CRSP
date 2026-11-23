import os, signal, subprocess, time
if os.path.exists("/content/cspd/exp10b.pid"):
    old = open("/content/cspd/exp10b.pid").read().strip()
    if old.isdigit():
        try: os.killpg(int(old), signal.SIGKILL)
        except Exception: pass
        time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp10b.log", "w") as f:
    f.write("launching exp10b...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp10b_bg.sh"],
    stdout=open("/content/cspd/exp10b.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp10b.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp10b PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
