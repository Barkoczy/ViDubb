# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ViDubb is an AI-powered video dubbing solution that provides voice cloning, multilingual translation, lip-sync synchronization, and background sound preservation. The system processes videos through speaker diarization, transcription, translation, voice synthesis, and optional lip synchronization.

## Core Architecture

### Main Components

1. **inference.py** - Command-line interface and main processing pipeline
2. **app.py** - Application entry point with logging setup
3. **gradio_interface.py** - Web interface using Gradio
4. **video_dubbing.py** - Core VideoDubbing class (modularized version)
5. **installer.py** - Package installation utilities
6. **tools/utils.py** - Utility functions for video/audio processing

### Processing Pipeline

The dubbing process follows this sequence:
1. **Speaker Diarization** - Identifies and separates speakers using pyannote.audio
2. **Audio Extraction** - Extracts audio segments for each speaker
3. **Transcription** - Uses Whisper for speech-to-text with word timestamps
4. **Translation** - MarianMT or Groq API for text translation
5. **Emotion Analysis** - SpeechBrain for emotion recognition
6. **Voice Synthesis** - XTTS v2 for voice cloning and generation
7. **Lip Sync (Optional)** - Wav2Lip for facial animation matching
8. **Audio Mixing** - Combines synthesized audio with background sounds

### Key Dependencies

- **AI/ML**: transformers, TTS, speechbrain, pyannote.audio, faster-whisper
- **Audio/Video**: pydub, opencv-python, ffmpeg-python, audio-separator
- **Web Interface**: gradio
- **APIs**: groq (optional), huggingface_hub

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -n "vidubbtest" python=3.10.14 ipython
conda activate vidubbtest

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FFmpeg
sudo apt-get install ffmpeg
```

### Model Downloads

```bash
# Download Wav2Lip models (required for lip sync)
wget 'https://github.com/medahmedkrichen/ViDubb/releases/download/weights2/wav2lip_gan.1.1.pth' -O 'Wav2Lip/wav2lip_gan.pth'
wget "https://github.com/medahmedkrichen/ViDubb/releases/download/weights1/s3fd-619a316812.1.1.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
```

### Running the Application

```bash
# Web interface
python app.py

# Command line interface
python inference.py --yt_url "https://youtube.com/watch?v=..." --source_language "en" --target_language "fr" --LipSync True --Bg_sound True

# Local video file
python inference.py --video_url "path/to/video.mp4" --source_language "en" --target_language "fr"
```

### Testing

No specific test framework is configured. Manual testing involves:
1. Processing sample videos with different language pairs
2. Verifying audio quality and lip sync accuracy
3. Testing with/without background sound preservation

## Configuration

### Environment Variables (.env file)

```bash
# Required for speaker diarization
HF_TOKEN="your_huggingface_token"

# Optional for enhanced translation
Groq_TOKEN="your_groq_token"
```

### Language Support

Supported languages are defined in `gradio_interface.py`:
- English, Spanish, French, German, Italian, Turkish, Russian
- Dutch, Czech, Arabic, Chinese (Simplified), Japanese, Korean, Hindi, Hungarian

## Key Processing Parameters

- **whisper_model**: "tiny", "base", "small", "medium", "large" (default: "medium")
- **LipSync**: Enable/disable Wav2Lip facial animation (computationally intensive)
- **Bg_sound**: Preserve background audio (may introduce noise)
- **Voice_denoising**: Clean audio output (default: enabled)

## File Structure

### Input/Output Directories

- `audio/` - Temporary audio processing files
- `results/` - Final output videos
- `speakers_audio/` - Individual speaker audio segments
- `speakers_image/` - Speaker facial images (for lip sync)
- `audio_chunks/` - TTS-generated audio segments
- `su_audio_chunks/` - Time-synchronized audio chunks

### Important Files

- `.env` - API tokens and configuration
- `frame_per_speaker.json` - Frame-to-speaker mapping for lip sync
- `Wav2Lip/` - Submodule for lip synchronization

## Performance Considerations

- CUDA GPU recommended for faster processing
- Large video files require significant memory
- Lip sync processing is memory and time intensive
- Background sound preservation may affect audio quality