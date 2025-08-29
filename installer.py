import os
import logging
import sys
import importlib
import warnings

# Apply ml_dtypes fix before any other imports
try:
    import ml_dtypes_fix
except ImportError:
    pass


# Fix NVML by setting CUDA library paths for PyTorch
nvidia_lib_paths = [
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu", 
    "/usr/local/cuda/lib64"
]

current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_path_updated = False

for nvidia_lib_path in nvidia_lib_paths:
    if os.path.exists(nvidia_lib_path) and nvidia_lib_path not in current_ld_path:
        if ld_path_updated:
            os.environ['LD_LIBRARY_PATH'] = f"{nvidia_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
        else:
            os.environ['LD_LIBRARY_PATH'] = f"{nvidia_lib_path}:{current_ld_path}"
            ld_path_updated = True

# Set CUDA paths and environment for proper GPU support
os.environ['CUDA_HOME'] = '/usr'
os.environ['CUDA_PATH'] = '/usr'
os.environ['CUDA_ROOT'] = '/usr'
os.environ['PATH'] = f"/usr/bin:{os.environ.get('PATH', '')}"
os.environ['LD_LIBRARY_PATH'] = f"/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Force PyTorch to find and use NVIDIA libraries
os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# NVML is now properly working with nvidia-driver-575
print("✓ NVIDIA driver 575 provides working NVML support")

logger = logging.getLogger(__name__)

def install_if_not_installed(import_name, install_command):
    logger.info(f"Checking if {import_name} is installed...")
    try:
        importlib.import_module(import_name)
        logger.info(f"{import_name} is already installed")
    except ImportError:
        logger.warning(f"{import_name} not found, installing with: {install_command}")
        result = os.system(f"{install_command} > /dev/null 2>&1")
        if result == 0:
            logger.info(f"Successfully installed {import_name}")
        else:
            logger.error(f"Failed to install {import_name} with exit code: {result}")
    except Exception as e:
        logger.error(f"Unexpected error while checking {import_name}: {str(e)}")
        raise

def install_required_packages():
    """Install all required packages for ViDubb"""
    logger.info("Installing required packages...")
    
    # Fix NVML first
    install_if_not_installed('pynvml', 'pip install pynvml==12.0.0')
    
    install_if_not_installed('google.protobuf', 'pip install protobuf==4.25.8')
    install_if_not_installed('spacy', 'pip install spacy==3.8.2')
    install_if_not_installed('TTS', 'pip install --no-deps TTS==0.21.0')
    install_if_not_installed('packaging', 'pip install packaging==20.9')
    install_if_not_installed('whisper', 'pip install openai-whisper==20240930')
    install_if_not_installed('deepface', 'pip install deepface==0.0.93')
    
    # Force numpy version
    os.system('pip install numpy==1.26.4 > /dev/null 2>&1')
    
    logger.info("Package installation complete")