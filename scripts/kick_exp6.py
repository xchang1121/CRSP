import os, subprocess, time
subprocess.run(["pkill", "-9", "-f", "exp6_teacherfree_input_sp.py"], check=False)
time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/exp6.log", "w")
proc = subprocess.Popen(
    ["bash", "/content/run_exp6_bg.sh"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
open("/content/cspd/exp6.pid", "w").write(str(proc.pid))
print("KILLED_OLD_AND_STARTED_EXP6_PID:", proc.pid)
