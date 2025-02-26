# ml_config.py - Complete ML Environment Configuration
# Import this file at the start of notebooks/sessions
# Usage: import ml_config

import os
import sys
import json
from pathlib import Path

# Base paths - adjust these according to your environment
LOCAL_MODELS_PATH = '/opt/ml/models'
LOCAL_DATASETS_PATH = '/opt/ml/datasets'
LOCAL_CACHE_PATH = '/opt/ml/cache'
LOCAL_PACKAGES_PATH = '/opt/ml/packages'
LOCAL_CONFIG_PATH = '/opt/ml/config'

def configure_ml_environment():
    """Configure all ML frameworks for offline/local usage"""
    # Ensure directories exist
    for path in [LOCAL_MODELS_PATH, LOCAL_DATASETS_PATH, LOCAL_CACHE_PATH, 
                LOCAL_PACKAGES_PATH, LOCAL_CONFIG_PATH]:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            os.environ[path.split('/')[-1].upper() + '_PATH'] = path
        except PermissionError:
            # Still set the environment variable even if we can't create the directory
            os.environ[path.split('/')[-1].upper() + '_PATH'] = path
            pass
    
    # Add local packages to Python path
    if LOCAL_PACKAGES_PATH not in sys.path:
        sys.path.append(LOCAL_PACKAGES_PATH)
    
    # Dictionary of all environment variables to set
    env_vars = {
        #=== Package and Dependency Management ===
        # PyPI Configuration
        'PIP_INDEX_URL': 'http://your-local-pypi-mirror/simple',
        'PIP_TRUSTED_HOST': 'your-local-pypi-mirror',
        'PIP_NO_CACHE_DIR': 'false',
        'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        
        #=== Hugging Face Ecosystem ===
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'HF_HOME': os.path.join(LOCAL_CACHE_PATH, 'huggingface'),
        'TRANSFORMERS_CACHE': os.path.join(LOCAL_MODELS_PATH, 'transformers'),
        'HF_DATASETS_CACHE': os.path.join(LOCAL_DATASETS_PATH, 'huggingface'),
        'HUGGINGFACE_HUB_CACHE': os.path.join(LOCAL_CACHE_PATH, 'huggingface/hub'),
        'HF_TRANSFER_DISABLE': '1',  # Disable accelerated downloads
        'TOKENIZERS_PARALLELISM': 'true',
        'BITSANDBYTES_NOWELCOME': '1',  # Disable welcome message
        
        #=== PyTorch Ecosystem ===
        'TORCH_HOME': os.path.join(LOCAL_MODELS_PATH, 'torch'),
        'TORCH_DATA_PATH': os.path.join(LOCAL_DATASETS_PATH, 'torch'),
        'PYTORCH_UPDATE_CHECK': '0',
        'TORCH_EXTENSIONS_DIR': os.path.join(LOCAL_CACHE_PATH, 'torch_extensions'),
        'TORCH_CUDNN_HOME': os.path.join(LOCAL_CACHE_PATH, 'cudnn'),
        
        #=== TensorFlow Ecosystem ===
        'KERAS_HOME': os.path.join(LOCAL_CACHE_PATH, 'keras'),
        'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TF verbosity
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_USE_LEGACY_KERAS': '0',  # Use Keras integrated in TF
        'TF_ENABLE_ONEDNN_OPTS': '0',
        
        #=== JAX Ecosystem ===
        'JAX_PLATFORMS': 'cpu,gpu',
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/usr/local/cuda',
        'JAX_ENABLE_X64': 'true',
        'JAX_TRACEBACK_FILTERING': 'off',
        
        #=== ONNX Runtime ===
        'ORT_TENSORRT_CACHE_PATH': os.path.join(LOCAL_CACHE_PATH, 'tensorrt'),
        'ORT_TENSORRT_ENGINE_CACHE_ENABLE': '1',
        'ORT_TENSORRT_FP16_ENABLE': '1',
        
        #=== Triton Inference Server ===
        'TRITON_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'triton'),
        'TRITON_KERNEL_PATH': os.path.join(LOCAL_MODELS_PATH, 'triton_kernels'),
        'TRITON_SKIP_UPDATE_CHECK': '1',
        'TRITON_SERVER_URL': 'localhost:8000',
        
        #=== LLM Tools ===
        # LiteLLM
        'LITELLM_CACHE_PATH': os.path.join(LOCAL_CACHE_PATH, 'litellm'),
        'LITELLM_MODEL_PATH': os.path.join(LOCAL_MODELS_PATH, 'litellm'),
        'LITELLM_TELEMETRY': 'false',
        'LITELLM_CONFIG_PATH': os.path.join(LOCAL_CONFIG_PATH, 'litellm_config.yaml'),
        
        # Text Generation Inference
        'TGI_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'tgi'),
        'TGI_OFFLINE': '1',
        
        # vLLM
        'VLLM_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'vllm'),
        'VLLM_MODEL_DIR': os.path.join(LOCAL_MODELS_PATH, 'vllm'),
        
        #=== Distributed Training ===
        # DeepSpeed
        'DEEPSPEED_CONFIG_PATH': os.path.join(LOCAL_CONFIG_PATH, 'deepspeed_config.json'),
        'DEEPSPEED_CACHE_PATH': os.path.join(LOCAL_CACHE_PATH, 'deepspeed'),
        'CUDA_CACHE_DISABLE': '0',
        'CUDA_MODULE_LOADING': 'LAZY',
        
        # Ray
        'RAY_HOME': os.path.join(LOCAL_CACHE_PATH, 'ray'),
        'RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE': '1',
        'RAY_DISABLE_DOCKER_CPU_WARNING': '1',
        
        #=== Data Science & ML Ops ===
        # MLflow
        'MLFLOW_TRACKING_URI': f'file://{os.path.join(LOCAL_MODELS_PATH, "mlflow")}',
        'MLFLOW_REGISTRY_URI': f'file://{os.path.join(LOCAL_MODELS_PATH, "mlflow/registry")}',
        'MLFLOW_TRACKING_INSECURE_TLS': 'true',
        
        # Weights & Biases
        'WANDB_MODE': 'offline',
        'WANDB_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'wandb'),
        'WANDB_SILENT': 'true',
        
        # DVC (Data Version Control)
        'DVC_DIR': os.path.join(LOCAL_CACHE_PATH, 'dvc'),
        'DVC_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'dvc/cache'),
        'DVC_NO_ANALYTICS': '1',
        
        #=== Data Science Libraries ===
        # Scikit-learn
        'SKLEARN_DATA_HOME': os.path.join(LOCAL_DATASETS_PATH, 'scikit-learn'),
        'JOBLIB_CACHE_DIR': os.path.join(LOCAL_CACHE_PATH, 'joblib'),
        'JOBLIB_TEMP_FOLDER': os.path.join(LOCAL_CACHE_PATH, 'joblib_temp'),
        
        # SciPy & NumPy
        'NPY_DISTUTILS_APPEND_FLAGS': '1',
        'OPENBLAS_NUM_THREADS': '4',  # Adjust based on system
        'MKL_NUM_THREADS': '4',       # Adjust based on system
        'OMP_NUM_THREADS': '4',       # Adjust based on system
        
        #=== GPU Configuration ===
        'CUDA_CACHE_PATH': os.path.join(LOCAL_CACHE_PATH, 'cuda'),
        'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility',
        'NVIDIA_VISIBLE_DEVICES': 'all',
        
        #=== Other Library-specific settings ===
        'TOKENIZERS_PARALLELISM': 'true',
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONHASHSEED': '42',  # For reproducibility
        'MPLCONFIGDIR': os.path.join(LOCAL_CACHE_PATH, 'matplotlib')
    }
    
    # Set environment variables
    for name, value in env_vars.items():
        os.environ[name] = value
    
    # Create default config files if they don't exist
    config_files = {
        'litellm_config.yaml': 'model_configs: {}',
        'deepspeed_config.json': '{}'
    }
    
    for filename, content in config_files.items():
        try:
            filepath = os.path.join(LOCAL_CONFIG_PATH, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    f.write(content)
        except (PermissionError, IOError):
            # Continue even if we can't create the config files
            pass
    
    # Save current environment to a JSON file for inspection
    try:
        env_file = os.path.join(LOCAL_CONFIG_PATH, 'current_env.json')
        with open(env_file, 'w') as f:
            json.dump({k: v for k, v in dict(os.environ).items() if not k.startswith('_')}, f, indent=2)
    except (PermissionError, IOError):
        # Continue even if we can't save the environment
        pass
    
    return True

def verify_configuration():
    """Verify that the ML environment is correctly configured"""
    # Check critical paths
    paths_to_check = [
        LOCAL_MODELS_PATH,
        LOCAL_DATASETS_PATH,
        LOCAL_CACHE_PATH,
        LOCAL_PACKAGES_PATH,
        LOCAL_CONFIG_PATH
    ]
    
    missing_paths = [p for p in paths_to_check if not os.path.exists(p)]
    
    # Check critical environment variables
    critical_vars = [
        "PIP_INDEX_URL",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TORCH_HOME",
        "KERAS_HOME",
        "MLFLOW_TRACKING_URI",
        "WANDB_MODE"
    ]
    
    missing_vars = [v for v in critical_vars if v not in os.environ]
    
    return not missing_paths and not missing_vars

# Auto-execute on import
configure_ml_environment()
verify_configuration()
