"""Patched overnight runner - applies runtime fixes before import."""

import importlib

# Patch docker_runner before importing
spec = importlib.util.spec_from_file_location("docker_runner", "docker_runner.py")
mod = importlib.util.module_from_spec(spec)
with open("docker_runner.py", "r") as f:
    src = f.read()
src = src.replace("MIN_VRAM_FREE_MB = 15000", "MIN_VRAM_FREE_MB = 6000")
src = src.replace(
    'f"-v /tmp/{self.train_script}:{CONTAINER_WORKDIR}/{self.train_script}:ro "',
    'f"-v C:/tmp/{self.train_script}:{CONTAINER_WORKDIR}/{self.train_script}:ro "',
)
with open("docker_runner.py", "w") as f:
    f.write(src)

# Patch run_overnight default chain
with open("run_overnight.py", "r") as f:
    src2 = f.read()
src2 = src2.replace('args.fallback_chain or ["gemini", "gemini", "kimi"]', "args.fallback_chain")
with open("run_overnight.py", "w") as f:
    f.write(src2)

# Now run the actual overnight script
exec(open("run_overnight.py").read())
