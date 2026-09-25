import platform, shutil, subprocess, sys

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=True)
        return (r.stdout.strip() + " " + r.stderr.strip()).strip()[:500]
    except Exception as e:
        return f"ERR: {e}"

print("== ENV CHECK ==")
print("python:", sys.version.split()[0])
print("pip:", run("pip --version"))
print("git:", run("git --version"))
print("winget:", run("winget --version")[:100])
print("ollama:", run("ollama --version"))
print("cmake:", run("cmake --version")[:100])
print("nvidia-smi:", run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")[:300])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch: NOT INSTALLED")
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception:
    print("transformers: NOT INSTALLED")
