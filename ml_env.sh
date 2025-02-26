#!/bin/bash
# Sets ML framework environment variables for offline usage

# Base paths
export LOCAL_MODELS_PATH="/opt/ml/models"
export LOCAL_DATASETS_PATH="/opt/ml/datasets"
export LOCAL_CACHE_PATH="/opt/ml/cache"
export LOCAL_PACKAGES_PATH="/opt/ml/packages"
export LOCAL_CONFIG_PATH="/opt/ml/config"

# Hugging Face Ecosystem
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_HOME="${LOCAL_CACHE_PATH}/huggingface"
export TRANSFORMERS_CACHE="${LOCAL_MODELS_PATH}/transformers"
export HF_DATASETS_CACHE="${LOCAL_DATASETS_PATH}/huggingface"
export HUGGINGFACE_HUB_CACHE="${LOCAL_CACHE_PATH}/huggingface/hub"
export HF_TRANSFER_DISABLE="1"
export TOKENIZERS_PARALLELISM="true"
export BITSANDBYTES_NOWELCOME="1"

# PyTorch Ecosystem
export TORCH_HOME="${LOCAL_MODELS_PATH}/torch"
export TORCH_DATA_PATH="${LOCAL_DATASETS_PATH}/torch"
export PYTORCH_UPDATE_CHECK="0"
export TORCH_EXTENSIONS_DIR="${LOCAL_CACHE_PATH}/torch_extensions"
export TORCH_CUDNN_HOME="${LOCAL_CACHE_PATH}/cudnn"

# TensorFlow Ecosystem
export KERAS_HOME="${LOCAL_CACHE_PATH}/keras"
export TF_CPP_MIN_LOG_LEVEL="2"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_USE_LEGACY_KERAS="0"
export TF_ENABLE_ONEDNN_OPTS="0"

# JAX Ecosystem
export JAX_PLATFORMS="cpu,gpu"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export JAX_ENABLE_X64="true"
export JAX_TRACEBACK_FILTERING="off"

# ONNX Runtime
export ORT_TENSORRT_CACHE_PATH="${LOCAL_CACHE_PATH}/tensorrt"
export ORT_TENSORRT_ENGINE_CACHE_ENABLE="1"
export ORT_TENSORRT_FP16_ENABLE="1"

# Triton Inference Server
export TRITON_CACHE_DIR="${LOCAL_CACHE_PATH}/triton"
export TRITON_KERNEL_PATH="${LOCAL_MODELS_PATH}/triton_kernels"
export TRITON_SKIP_UPDATE_CHECK="1"
export TRITON_SERVER_URL="localhost:8000"

# LiteLLM
export LITELLM_CACHE_PATH="${LOCAL_CACHE_PATH}/litellm"
export LITELLM_MODEL_PATH="${LOCAL_MODELS_PATH}/litellm"
export LITELLM_TELEMETRY="false"
export LITELLM_CONFIG_PATH="${LOCAL_CONFIG_PATH}/litellm_config.yaml"

# Text Generation Inference
export TGI_CACHE_DIR="${LOCAL_CACHE_PATH}/tgi"
export TGI_OFFLINE="1"

# vLLM
export VLLM_CACHE_DIR="${LOCAL_CACHE_PATH}/vllm"
export VLLM_MODEL_DIR="${LOCAL_MODELS_PATH}/vllm"

# DeepSpeed
export DEEPSPEED_CONFIG_PATH="${LOCAL_CONFIG_PATH}/deepspeed_config.json"
export DEEPSPEED_CACHE_PATH="${LOCAL_CACHE_PATH}/deepspeed"
export CUDA_CACHE_DISABLE="0"
export CUDA_MODULE_LOADING="LAZY"

# Ray
export RAY_HOME="${LOCAL_CACHE_PATH}/ray"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="1"
export RAY_DISABLE_DOCKER_CPU_WARNING="1"

# MLflow
export MLFLOW_TRACKING_URI="file://${LOCAL_MODELS_PATH}/mlflow"
export MLFLOW_REGISTRY_URI="file://${LOCAL_MODELS_PATH}/mlflow/registry"
export MLFLOW_TRACKING_INSECURE_TLS="true"

# Weights & Biases
export WANDB_MODE="offline"
export WANDB_CACHE_DIR="${LOCAL_CACHE_PATH}/wandb"
export WANDB_SILENT="true"

# DVC (Data Version Control)
export DVC_DIR="${LOCAL_CACHE_PATH}/dvc"
export DVC_CACHE_DIR="${LOCAL_CACHE_PATH}/dvc/cache"
export DVC_NO_ANALYTICS="1"

# Scikit-learn
export SKLEARN_DATA_HOME="${LOCAL_DATASETS_PATH}/scikit-learn"
export JOBLIB_CACHE_DIR="${LOCAL_CACHE_PATH}/joblib"
export JOBLIB_TEMP_FOLDER="${LOCAL_CACHE_PATH}/joblib_temp"

# SciPy & NumPy
export NPY_DISTUTILS_APPEND_FLAGS="1"
export OPENBLAS_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export OMP_NUM_THREADS="4"

# GPU Configuration
export CUDA_CACHE_PATH="${LOCAL_CACHE_PATH}/cuda"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"
export NVIDIA_VISIBLE_DEVICES="all"

# Other settings
export TOKENIZERS_PARALLELISM="true"
export PYTHONIOENCODING="utf-8"
export PYTHONHASHSEED="42"
export MPLCONFIGDIR="${LOCAL_CACHE_PATH}/matplotlib"

# Add local packages to Python path
export PYTHONPATH="${LOCAL_PACKAGES_PATH}:${PYTHONPATH}"

echo "ML environment variables set for offline use."
