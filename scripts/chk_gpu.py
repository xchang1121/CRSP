import subprocess
print(subprocess.check_output(["nvidia-smi"], text=True))
