# Import ML configuration 
import sys
sys.path.append('/opt/ml')
import ml_config

# Add to /etc/jupyter/jupyter_notebook_config.py or similar file

c.ServerApp.terminals_enabled = True
c.ServerApp.ip = '0.0.0.0'

# Custom welcome message
c.ServerApp.welcome_banner = """
============================================================
Welcome to JupyterHub ML Environment
============================================================

This JupyterHub instance is configured for offline ML development.

To enable ML environment variables in your notebook, 
add these lines at the top of your notebook:

    import sys
    sys.path.append('/opt/ml')
    import ml_config

This will configure:
- Offline mode for HuggingFace, PyTorch, and TensorFlow
- Local cache and model directories
- Optimized ML paths and configurations
- Local package mirrors

All ML libraries are pre-configured to work in an offline environment.
============================================================
"""

# Message will also show in notebook dashboard
c.FileContentsManager.root_dir = '/'

# You can also create a custom message in the landing page
