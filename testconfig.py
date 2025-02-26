# Test environment configuration script
# Run in a notebook cell or as a Python script

import os
import sys
import json
from pprint import pprint

def check_environment():
    """Check if ML offline configuration has been correctly applied"""
    
    print("=== CHECKING ENVIRONMENT CONFIGURATION ===\n")
    
    # 1. Check ML Framework paths
    print("ML PATHS CONFIGURED:")
    ml_env_vars = [
        # PyPI
        "PIP_INDEX_URL", "PIP_TRUSTED_HOST",
        # Hugging Face
        "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HOME", "TRANSFORMERS_CACHE",
        # PyTorch
        "TORCH_HOME", "PYTORCH_UPDATE_CHECK",
        # TensorFlow
        "KERAS_HOME", "TF_CPP_MIN_LOG_LEVEL",
        # Triton
        "TRITON_CACHE_DIR", "TRITON_SKIP_UPDATE_CHECK", 
        # LiteLLM
        "LITELLM_CACHE_PATH", "LITELLM_TELEMETRY",
        # Others
        "WANDB_MODE", "MLFLOW_TRACKING_URI"
    ]
    
    env_status = {}
    for var in ml_env_vars:
        env_status[var] = os.environ.get(var, "NOT SET")
    
    # Pretty print important ML environment variables
    pprint(env_status)
    
    # 2. Check Python paths (to verify local packages are in path)
    print("\nPYTHON PATH:")
    for path in sys.path:
        if "/opt/ml" in path:
            print(f"✓ ML Path found: {path}")
    
    # 3. Check directory existence
    print("\nCHECKING DIRECTORIES:")
    paths_to_check = [
        "/opt/ml/models",
        "/opt/ml/datasets", 
        "/opt/ml/cache",
        "/opt/ml/packages",
        "/opt/ml/mlflow",
        "/opt/ml/config"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✓ Directory exists: {path}")
        else:
            print(f"✗ Missing directory: {path}")
    
    # 4. Check import system to make sure it's not reaching external sites
    print("\nIMPORT SYSTEM TEST:")
    try:
        import site
        print(f"Site packages directory: {site.getsitepackages()}")
    except Exception as e:
        print(f"Error checking site packages: {e}")
        
    # 5. Test pip configuration
    print("\nPIP CONFIGURATION:")
    try:
        import pip
        pip_version = pip.__version__
        print(f"Pip version: {pip_version}")
        
        # Get pip config without actually running pip
        if hasattr(pip, "_internal") and hasattr(pip._internal, "configuration"):
            pip_config = pip._internal.configuration.Configuration(isolated=False)
            config_items = pip_config.get_values_override()
            if config_items:
                print("Pip configuration values:")
                for key, value in config_items:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error checking pip configuration: {e}")

    print("\n=== ENVIRONMENT CHECK COMPLETE ===")

if __name__ == "__main__":
    check_environment()
