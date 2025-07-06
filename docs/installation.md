# Installation Guide

This guide provides detailed instructions for installing and setting up the MSA-T OSV framework.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB+ recommended for training)
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for training)

### CUDA Requirements (for GPU training)

- **CUDA**: 11.0 or higher
- **cuDNN**: 8.0 or higher
- **GPU Memory**: At least 4GB VRAM (8GB+ recommended)

## Installation Methods

### Method 1: Install from Source (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd msa_t_osv
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using conda
   conda create -n msa_t_osv python=3.10
   conda activate msa_t_osv
   
   # Or using venv
   python -m venv msa_t_osv_env
   source msa_t_osv_env/bin/activate  # On Windows: msa_t_osv_env\Scripts\activate
   ```

3. **Install PyTorch** (with CUDA support if available):
   ```bash
   # For CUDA 11.8
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install the package**:
   ```bash
   pip install -e .
   ```

### Method 2: Install with pip

```bash
pip install msa-t-osv
```

### Method 3: Using Docker

1. **Build the Docker image**:
   ```bash
   docker build -t msa-t-osv .
   ```

2. **Run the container**:
   ```bash
   docker run -it --gpus all -v $(pwd):/app msa-t-osv
   ```

## Verification

### Basic Verification

1. **Check installation**:
   ```bash
   python -c "import msa_t_osv; print('MSA-T OSV installed successfully!')"
   ```

2. **Run basic example**:
   ```bash
   python examples/basic_usage.py
   ```

3. **Check GPU availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU count: {torch.cuda.device_count()}")
   ```

### Advanced Verification

1. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

2. **Check model creation**:
   ```python
   from msa_t_osv.models import MSATOSVModel
   import yaml
   
   # Load sample config
   with open('configs/cedar.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Create model
   model = MSATOSVModel(config)
   print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
   ```

## Development Setup

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Setup Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

### Code Quality Tools

The project uses several code quality tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run them manually:
```bash
# Format code
black msa_t_osv/

# Sort imports
isort msa_t_osv/

# Run linter
flake8 msa_t_osv/

# Type checking
mypy msa_t_osv/
```

## Troubleshooting

### Common Issues

#### 1. CUDA Installation Issues

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Reinstall PyTorch with correct CUDA version
- Ensure NVIDIA drivers are up to date

#### 2. Memory Issues

**Problem**: CUDA out of memory errors

**Solutions**:
- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training
- Use smaller model configurations

#### 3. Import Errors

**Problem**: Module not found errors

**Solutions**:
- Ensure virtual environment is activated
- Reinstall the package: `pip install -e .`
- Check Python path: `python -c "import sys; print(sys.path)"`

#### 4. Dataset Loading Issues

**Problem**: Dataset not found or loading errors

**Solutions**:
- Verify dataset paths in configuration files
- Check file permissions
- Ensure dataset format is correct
- Download datasets if not present

### Platform-Specific Issues

#### Windows

- Use WSL2 for better compatibility
- Install Visual Studio Build Tools for C++ extensions
- Use conda instead of pip for better package management

#### macOS

- Install Xcode Command Line Tools: `xcode-select --install`
- Use conda for M1/M2 Macs
- Install PyTorch with MPS support for Apple Silicon

#### Linux

- Install system dependencies: `sudo apt-get install build-essential`
- Install CUDA toolkit from NVIDIA website
- Set environment variables in `~/.bashrc`

### Performance Optimization

#### GPU Optimization

1. **Enable mixed precision**:
   ```yaml
   training:
     use_amp: true
   ```

2. **Optimize data loading**:
   ```yaml
   training:
     num_workers: 4
     pin_memory: true
   ```

3. **Use gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 2
   ```

#### Memory Optimization

1. **Reduce batch size**
2. **Use gradient checkpointing**
3. **Enable memory efficient attention**

## Environment Variables

Set these environment variables for optimal performance:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# PyTorch settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Next Steps

After successful installation:

1. **Download datasets**: See [Dataset Guide](datasets.md)
2. **Configure models**: See [Configuration Guide](configuration.md)
3. **Run training**: See [Training Guide](training.md)
4. **Evaluate models**: See [Evaluation Guide](evaluation.md)

## Support

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search existing [GitHub issues](https://github.com/your-repo/issues)
3. Create a new issue with detailed information
4. Contact the maintainers

## Contributing

See [Contributing Guidelines](CONTRIBUTING.md) for development setup and contribution guidelines. 