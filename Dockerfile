# ML Framework offline setup
RUN mkdir -p /opt/ml/{models,datasets,cache,packages,mlflow,config} \
    /opt/ml/cache/{triton,litellm,tensorrt,deepspeed,ray,dvc,dvc/cache} \
    /opt/ml/models/{triton_kernels,litellm}

# Create default config files
RUN echo "{}" > /opt/ml/config/deepspeed_config.json && \
    echo "model_configs: {}" > /opt/ml/config/litellm_config.yaml

# Copy ML offline configuration
COPY ml_offline_config.py /usr/local/etc/jupyter/
RUN chmod 644 /usr/local/etc/jupyter/ml_offline_config.py

# Set up pip configuration
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = http://your-local-pypi-mirror/simple" >> /root/.pip/pip.conf && \
    echo "trusted-host = your-local-pypi-mirror" >> /root/.pip/pip.conf

# Add import of ML config to IPython startup
RUN mkdir -p /root/.ipython/profile_default/startup && \
    echo "import sys; sys.path.append('/usr/local/etc/jupyter')" > /root/.ipython/profile_default/startup/00-ml-config.py && \
    echo "import ml_offline_config" >> /root/.ipython/profile_default/startup/00-ml-config.py

# Set required environment variables persistently
ENV PIP_INDEX_URL=http://your-local-pypi-mirror/simple \
    PIP_TRUSTED_HOST=your-local-pypi-mirror \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTORCH_UPDATE_CHECK=0 \
    TF_CPP_MIN_LOG_LEVEL=2

# Set correct permissions
RUN chmod -R 755 /opt/ml && \
    chown -R jupyter:jupyter /opt/ml  # Adjust user if needed
