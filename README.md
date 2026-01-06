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

## Requirements
```
(tested on)
1. python == 3.13
2. pytorch == 2.1.1
3. torchvision == 0.16.1

```

## Notes
```
# 1. I just have notebooks folder to verify all the classes that I create of the model, dataset, evaluation metrics.

```

## Model Architecture
![alt text](media/model_infographics.png "Model Infographics")