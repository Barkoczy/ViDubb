import os
import logging
import sys
from datetime import datetime

# Fix TensorFlow warnings but keep CUDA enabled
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, keep ERROR and FATAL
# Don't disable CUDA - we need it for performance!
# Legacy filesystem compatibility
os.environ['TF_ENABLE_LEGACY_FILESYSTEM'] = '1'

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('vidubb_detailed.log', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# No more warning suppression - fix root causes instead

logger.info("=== ViDubb Application Starting ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

# Install required packages FIRST before any imports
from installer import install_required_packages
install_required_packages()

# Import our modular components AFTER installation
from gradio_interface import launch_interface

# Import libraries with logging
logger.info("Starting library imports...")

# TensorFlow environment already configured above - no need to set again

try:
    logger.info("Importing audio processing libraries...")
    from pyannote.audio import Pipeline
    from audio_separator.separator import Separator
    import whisper
    logger.info("Audio libraries imported successfully")
except Exception as e:
    logger.error(f"Failed to import audio libraries: {str(e)}")
    raise

try:
    logger.info("Importing ML/AI libraries...")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        from transformers import MarianMTModel, MarianTokenizer
        from TTS.api import TTS
        from speechbrain.inference.interfaces import foreign_class
        from deepface import DeepFace
        import torch
    logger.info("ML/AI libraries imported successfully")
except Exception as e:
    logger.error(f"Failed to import ML/AI libraries: {str(e)}")
    raise

try:
    logger.info("Importing utility libraries...")
    from pydub import AudioSegment
    import shutil
    import subprocess
    import numpy as np
    import cv2
    import json
    import re
    from groq import Groq
    from IPython.display import HTML, Audio
    from base64 import b64decode
    from scipy.io.wavfile import read as wav_read
    import nltk
    from nltk import sent_tokenize
    from faster_whisper import WhisperModel
    logger.info("Utility libraries imported successfully")
except Exception as e:
    logger.error(f"Failed to import utility libraries: {str(e)}")
    raise

logger.info("All libraries imported successfully")
logger.info("=== Starting Application Interface ===")

# Launch the Gradio interface
if __name__ == "__main__":
    launch_interface()