import os, subprocess, time
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp9.log", "w") as f:
    f.write("launching exp9...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp9_bg.sh"],
    stdout=open("/content/cspd/exp9.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp9.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp9 PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
