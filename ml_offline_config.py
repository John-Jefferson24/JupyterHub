# ml_offline_config.py
# Handles ML framework configs and offline settings only
import os
import sys
from pathlib import Path

# Base paths - adjust according to your environment
LOCAL_MODELS_PATH = '/opt/ml/models'
LOCAL_DATASETS_PATH = '/opt/ml/datasets'
LOCAL_CACHE_PATH = '/opt/ml/cache'
LOCAL_PACKAGES_PATH = '/opt/ml/packages'

# Create directories if they don't exist
for path in [LOCAL_MODELS_PATH, LOCAL_DATASETS_PATH, LOCAL_CACHE_PATH, LOCAL_PACKAGES_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Add local packages to Python path
sys.path.append(LOCAL_PACKAGES_PATH)

# ============= Package Mirror Configuration =============
os.environ['PIP_INDEX_URL'] = 'http://your-local-pypi-mirror/simple'
os.environ['PIP_TRUSTED_HOST'] = 'your-local-pypi-mirror'

# ============= Hugging Face Configuration =============
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.path.join(LOCAL_CACHE_PATH, 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(LOCAL_MODELS_PATH, 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(LOCAL_DATASETS_PATH, 'huggingface')

# ============= PyTorch Configuration =============
os.environ['TORCH_HOME'] = os.path.join(LOCAL_MODELS_PATH, 'torch')
os.environ['TORCH_DATA_PATH'] = os.path.join(LOCAL_DATASETS_PATH, 'torch')
os.environ['PYTORCH_UPDATE_CHECK'] = '0'

# ============= TensorFlow Configuration =============
os.environ['KERAS_HOME'] = os.path.join(LOCAL_CACHE_PATH, 'keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ============= JAX Configuration =============
os.environ['JAX_PLATFORMS'] = 'cpu,gpu'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

# ============= Scipy/Numpy Configuration =============
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ============= MLflow Configuration =============
os.environ['MLFLOW_TRACKING_URI'] = 'file:///opt/ml/mlflow'
os.environ['MLFLOW_REGISTRY_URI'] = 'file:///opt/ml/mlflow/registry'

# ============= Weights & Biases Configuration =============
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_CACHE_DIR'] = os.path.join(LOCAL_CACHE_PATH, 'wandb')

# ============= Triton Configuration =============
os.environ['TRITON_CACHE_DIR'] = os.path.join(LOCAL_CACHE_PATH, 'triton')
os.environ['TRITON_KERNEL_PATH'] = os.path.join(LOCAL_MODELS_PATH, 'triton_kernels')
# Disable Triton auto-update
os.environ['TRITON_SKIP_UPDATE_CHECK'] = '1'
# Local Triton server settings (if running locally)
os.environ['TRITON_SERVER_URL'] = 'localhost:8000'

# ============= LiteLLM Configuration =============
os.environ['LITELLM_CACHE_PATH'] = os.path.join(LOCAL_CACHE_PATH, 'litellm')
os.environ['LITELLM_MODEL_PATH'] = os.path.join(LOCAL_MODELS_PATH, 'litellm')
# Disable telemetry
os.environ['LITELLM_TELEMETRY'] = 'false'
# Local model configs
os.environ['LITELLM_CONFIG_PATH'] = '/opt/ml/config/litellm_config.yaml'

# ============= ONNX Runtime Configuration =============
os.environ['ORT_TENSORRT_CACHE_PATH'] = os.path.join(LOCAL_CACHE_PATH, 'tensorrt')
os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'

# ============= DeepSpeed Configuration =============
os.environ['DEEPSPEED_CONFIG_PATH'] = '/opt/ml/config/deepspeed_config.json'
os.environ['DEEPSPEED_CACHE_PATH'] = os.path.join(LOCAL_CACHE_PATH, 'deepspeed')

# ============= Ray Configuration =============
os.environ['RAY_HOME'] = os.path.join(LOCAL_CACHE_PATH, 'ray')
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
os.environ['RAY_DISABLE_DOCKER_CPU_WARNING'] = '1'

# ============= DVC (Data Version Control) Configuration =============
os.environ['DVC_DIR'] = os.path.join(LOCAL_CACHE_PATH, 'dvc')
os.environ['DVC_CACHE_DIR'] = os.path.join(LOCAL_CACHE_PATH, 'dvc/cache')

# ============= Scikit-learn Configuration =============
os.environ['SKLEARN_DATA_HOME'] = os.path.join(LOCAL_DATASETS_PATH, 'scikit-learn')

# Verification function
def verify_offline_setup():
    """Verify that all necessary directories exist and permissions are correct"""
    paths_to_check = [
        LOCAL_MODELS_PATH,
        LOCAL_DATASETS_PATH,
        LOCAL_CACHE_PATH,
        LOCAL_PACKAGES_PATH,
        os.path.join(LOCAL_CACHE_PATH, 'triton'),
        os.path.join(LOCAL_MODELS_PATH, 'triton_kernels'),
        os.path.join(LOCAL_CACHE_PATH, 'litellm'),
        os.path.join(LOCAL_MODELS_PATH, 'litellm'),
        os.path.join(LOCAL_CACHE_PATH, 'tensorrt'),
        os.path.join(LOCAL_CACHE_PATH, 'deepspeed'),
        os.path.join(LOCAL_CACHE_PATH, 'ray'),
        os.path.join(LOCAL_CACHE_PATH, 'dvc'),
        os.path.join(LOCAL_CACHE_PATH, 'dvc/cache'),
        '/opt/ml/config'  # For configuration files
    ]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Warning: Required path does not exist: {path}")
        elif not os.access(path, os.W_OK):
            print(f"Warning: No write permission for path: {path}")
        else:
            print(f"Path verified: {path}")

if __name__ == "__main__":
    verify_offline_setup()