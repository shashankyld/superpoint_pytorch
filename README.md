# Setup 
```
# 1. Clone the repository
git clone git@github.com:shashankyld/superpoint_pytorch.git
cd superpoint_pytorch

# 2. Sync the environment (Creates .venv and installs all dependencies from uv.lock)
uv sync

# 3. Verify the installation
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Dev tools
```
# Check for errors
uv run ruff check .

# Formats the codebase
uv run ruff format .
```
