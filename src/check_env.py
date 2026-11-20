import subprocess, sys, torch

print("=== GPU ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / (1024**3):.1f} GB, BF16 supported: {torch.cuda.is_bf16_supported()}")

print("\n=== Disk & RAM ===")
print(subprocess.check_output(["df", "-h", "/content", "/"], text=True))
print(subprocess.check_output(["free", "-h"], text=True))

print("=== Packages ===")
for pkg in ["torch", "transformers", "datasets", "accelerate", "peft", "sympy", "vllm", "huggingface_hub"]:
    try:
        m = __import__(pkg)
        print(f"  {pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"  {pkg}: NOT installed")
