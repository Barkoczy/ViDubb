import os
import logging
import warnings
import json
import re
import shutil
import subprocess
import numpy as np
import cv2
import torch
from datetime import datetime

# Audio processing imports
from pyannote.audio import Pipeline
from audio_separator.separator import Separator
import whisper
from pydub import AudioSegment
from scipy.io.wavfile import read as wav_read

# ML/AI imports
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from speechbrain.inference.interfaces import foreign_class
from deepface import DeepFace
from groq import Groq

# Text processing
from nltk import sent_tokenize

# Whisper import
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class VideoDubbing:
    def __init__(
        self,
        Video_path,
        source_language,
        target_language,
        LipSync=True,
        Voice_denoising=True,
        whisper_model="medium",
        Context_translation="API code here",
        huggingface_auth_token="API code here",
    ):

        # Initialize instance logger
        self.logger = logging.getLogger(f"{__name__}.VideoDubbing")

        self.logger.info("=== Initializing VideoDubbing ===")
        self.logger.info(f"Video path: {Video_path}")
        self.logger.info(f"Source language: {source_language}")
        self.logger.info(f"Target language: {target_language}")
        self.logger.info(f"LipSync enabled: {LipSync}")
        self.logger.info(f"Voice denoising enabled: {Voice_denoising}")
        self.logger.info(f"Whisper model: {whisper_model}")

        self.Video_path = Video_path
        self.source_language = source_language
        self.target_language = target_language
        self.LipSync = LipSync
        self.Voice_denoising = Voice_denoising
        self.whisper_model = whisper_model
        self.Context_translation = Context_translation
        self.huggingface_auth_token = huggingface_auth_token

        # Setup directories with logging
        self.logger.info("Setting up directories...")
        try:
            if os.path.exists("audio"):
                self.logger.info("Removing existing audio directory")
                os.system("rm -r audio")
            os.system("mkdir audio")
            self.logger.info("Audio directory created successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup audio directory: {str(e)}")
            raise

        try:
            if os.path.exists("results"):
                self.logger.info("Removing existing results directory")
                os.system("rm -r results")
            os.system("mkdir results")
            self.logger.info("Results directory created successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup results directory: {str(e)}")
            raise

        # Device setup with logging
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        if device.type == "cuda":
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            )

        # Initialize the pre-trained speaker diarization pipeline
        self.logger.info("Initializing speaker diarization pipeline...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.huggingface_auth_token,
                ).to(device)
            self.logger.info("Speaker diarization pipeline loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load speaker diarization pipeline: {str(e)}")
            raise

        # Load the audio from the video file with error handling
        self.logger.info(f"Extracting audio from video: {self.Video_path}")
        try:
            # First try to extract audio with FFmpeg using error resilience
            extract_command = f"ffmpeg -err_detect ignore_err -fflags +igndts+ignidx -i '{self.Video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 -f wav -y audio/test0_temp.wav"
            self.logger.info(f"Running FFmpeg command: {extract_command}")
            result = subprocess.run(
                extract_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0 and os.path.exists("audio/test0_temp.wav"):
                self.logger.info("FFmpeg audio extraction successful")
                # If successful, load and re-export for consistency
                audio = AudioSegment.from_file("audio/test0_temp.wav", format="wav")
                audio.export("audio/test0.wav", format="wav")
                os.remove("audio/test0_temp.wav")
                self.logger.info("Audio processed and saved as test0.wav")
            else:
                self.logger.warning(
                    f"FFmpeg failed with return code: {result.returncode}"
                )
                self.logger.warning(f"FFmpeg stderr: {result.stderr}")
                self.logger.info("Trying fallback: direct pydub extraction")
                # Fallback: try with pydub directly
                audio = AudioSegment.from_file(self.Video_path, format="mp4")
                audio.export("audio/test0.wav", format="wav")
                self.logger.info("Fallback pydub extraction successful")
        except Exception as e:
            self.logger.error(f"Audio extraction failed with error: {str(e)}")
            self.logger.info("Creating silent audio as last resort")
            # Last resort: create silent audio
            audio = AudioSegment.silent(duration=10000)  # 10 seconds of silence
            audio.export("audio/test0.wav", format="wav")
            self.logger.warning("Using silent audio due to extraction failure")

        audio_file = "audio/test0.wav"

        # Apply the diarization pipeline on the audio file
        self.logger.info("Applying speaker diarization pipeline on audio file")
        try:
            diarization = pipeline(audio_file)
            self.logger.info("Speaker diarization completed successfully")
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {str(e)}")
            raise

        speakers_rolls = {}

        # Process the diarization results
        self.logger.info("Processing diarization results")
        speaker_count = 0
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = abs(speech_turn.end - speech_turn.start)
            if duration > 1.5:
                self.logger.info(
                    f"Speaker {speaker}: from {speech_turn.start:.2f}s to {speech_turn.end:.2f}s (duration: {duration:.2f}s)"
                )
                speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
                speaker_count += 1

        self.logger.info(
            f"Found {speaker_count} valid speech segments (>1.5s) from {len(list(diarization.itertracks()))} total segments"
        )
        self.logger.info(
            f"Total unique speakers detected: {len(set(speakers_rolls.values()))}"
        )

        # speakers_rolls = merge_overlapping_periods(speakers_rolls)

        if self.LipSync:
            self.logger.info("LipSync enabled - starting video processing")
            # Load the video file
            self.logger.info(f"Loading video file: {self.Video_path}")
            video = cv2.VideoCapture(self.Video_path)

            if not video.isOpened():
                self.logger.error(f"Failed to open video file: {self.Video_path}")
                raise Exception(f"Could not open video file: {self.Video_path}")

            # Get frames per second (FPS)
            fps = video.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Video FPS: {fps}")

            # Get total number of frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Total video frames: {total_frames}")

            video.release()
            self.logger.info("Video file closed after reading properties")

            frame_per_speaker = []
            self.logger.info("Mapping frames to speakers")

            for i in range(total_frames):
                time = i / round(fps)
                frame_speaker = get_speaker(time, speakers_rolls)
                frame_per_speaker.append(frame_speaker)
                if i % 1000 == 0:  # Log progress every 1000 frames
                    self.logger.info(
                        f"Processed {i}/{total_frames} frames ({(i/total_frames)*100:.1f}%)"
                    )

            self.logger.info(
                f"Frame-to-speaker mapping completed for {total_frames} frames"
            )

            self.logger.info("Setting up speakers_image directory")
            if os.path.exists("speakers_image"):
                self.logger.info("Removing existing speakers_image directory")
                os.system("rm -r speakers_image")

            self.logger.info("Creating new speakers_image directory")
            os.system("mkdir speakers_image")

            # Specify the video path and output folder
            output_folder = "speakers_image"
            self.logger.info(f"Extracting frames to {output_folder}")
            # Call the function
            try:
                extract_frames(self.Video_path, output_folder, speakers_rolls)
                self.logger.info("Frame extraction completed successfully")
            except Exception as e:
                self.logger.error(f"Frame extraction failed: {str(e)}")
                raise

            # Initialize the face detector
            self.logger.info("Initializing face detection system")
            haar_cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.logger.info(f"Loading Haar Cascade from: {haar_cascade_path}")

            # Load the pre-trained Haar Cascade model for face detection
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)

            if face_cascade.empty():
                self.logger.error("Failed to load Haar Cascade classifier")
                raise Exception("Could not load face detection model")

            self.logger.info("Face detection system initialized successfully")

            # Function to detect and crop faces

            # Path to the folder containing speaker images
            speaker_images_folder = "speakers_image"

            # Process face detection on extracted speaker images
            self.logger.info("Starting face detection on speaker images")
            processed_images = 0
            deleted_images = 0

            # Iterate through speaker subfolders
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(
                    speaker_images_folder, speaker_folder
                )

                if os.path.isdir(speaker_folder_path):
                    self.logger.info(f"Processing speaker folder: {speaker_folder}")
                    # Process each image in the speaker folder
                    for image_name in os.listdir(speaker_folder_path):
                        image_path = os.path.join(speaker_folder_path, image_name)
                        processed_images += 1

                        # Detect and crop faces from the image
                        if not detect_and_crop_faces(image_path, face_cascade):
                            # If no face is detected, delete the image
                            os.remove(image_path)
                            deleted_images += 1
                            self.logger.debug(
                                f"Deleted {image_path} - no face detected"
                            )
                        else:
                            self.logger.debug(
                                f"Face detected and cropped: {image_path}"
                            )

            self.logger.info(
                f"Face detection completed. Processed: {processed_images}, Deleted: {deleted_images}, Kept: {processed_images - deleted_images}"
            )

            # Extract most common faces for each speaker
            self.logger.info("Extracting most common face for each speaker")
            speaker_images_folder = "speakers_image"
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(
                    speaker_images_folder, speaker_folder
                )

                self.logger.info(f"Processing images in folder: {speaker_folder}")
                try:
                    extract_and_save_most_common_face(speaker_folder_path)
                    self.logger.info(
                        f"Successfully extracted most common face for {speaker_folder}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to extract most common face for {speaker_folder}: {str(e)}"
                    )

            # Clean up temporary image files, keep only max_image.jpg
            self.logger.info("Cleaning up temporary image files")
            cleanup_count = 0
            for root, dirs, files in os.walk(speaker_images_folder):
                for file in files:
                    # Check if the file is not 'max_image.jpg'
                    if file != "max_image.jpg":
                        # Construct full file path
                        file_path = os.path.join(root, file)
                        # Delete the file
                        os.remove(file_path)
                        cleanup_count += 1

            self.logger.info(f"Cleaned up {cleanup_count} temporary image files")

            # Save frame-to-speaker mapping and copy files to Wav2Lip directory
            self.logger.info("Saving frame-to-speaker mapping")
            try:
                with open("frame_per_speaker.json", "w") as f:
                    json.dump(frame_per_speaker, f)
                self.logger.info("Frame-to-speaker mapping saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save frame-to-speaker mapping: {str(e)}")
                raise

            # Copy files to Wav2Lip directory
            self.logger.info("Copying files to Wav2Lip directory")
            try:
                if os.path.exists("Wav2Lip/frame_per_speaker.json"):
                    os.remove("Wav2Lip/frame_per_speaker.json")
                shutil.copyfile(
                    "frame_per_speaker.json", "Wav2Lip/frame_per_speaker.json"
                )
                self.logger.info("Frame mapping file copied to Wav2Lip")

                if os.path.exists("Wav2Lip/speakers_image"):
                    shutil.rmtree("Wav2Lip/speakers_image")
                shutil.copytree("speakers_image", "Wav2Lip/speakers_image")
                self.logger.info("Speaker images copied to Wav2Lip")
            except Exception as e:
                self.logger.error(f"Failed to copy files to Wav2Lip: {str(e)}")
                raise

        ###############################################################################
        # Audio processing section
        ###############################################################################

        self.logger.info("Starting audio processing for individual speakers")
        if os.path.exists("speakers_audio"):
            self.logger.info("Removing existing speakers_audio directory")
            os.system("rm -r speakers_audio")

        self.logger.info("Creating speakers_audio directory")
        os.system("mkdir speakers_audio")

        speakers = set(list(speakers_rolls.values()))
        self.logger.info(f"Found {len(speakers)} unique speakers: {list(speakers)}")

        try:
            audio = AudioSegment.from_file(audio_file, format="wav")
            self.logger.info(f"Loaded audio file: {audio_file}")
        except Exception as e:
            self.logger.error(f"Failed to load audio file {audio_file}: {str(e)}")
            raise

        for speaker in speakers:
            self.logger.info(f"Processing audio segments for speaker: {speaker}")
            speaker_audio = AudioSegment.empty()
            segment_count = 0

            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(key[0]) * 1000
                    end = int(key[1]) * 1000

                    speaker_audio += audio[start:end]
                    segment_count += 1

            try:
                speaker_audio.export(f"speakers_audio/{speaker}.wav", format="wav")
                duration = len(speaker_audio) / 1000.0
                self.logger.info(
                    f"Exported {speaker}.wav: {segment_count} segments, {duration:.2f}s total"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to export audio for speaker {speaker}: {str(e)}"
                )
                raise

        most_occured_speaker = max(
            list(speakers_rolls.values()), key=list(speakers_rolls.values()).count
        )
        self.logger.info(f"Most frequently occurring speaker: {most_occured_speaker}")

        # Initialize Whisper model for transcription
        self.logger.info(f"Initializing Whisper model: {self.whisper_model}")
        try:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device for Whisper: {device_str}")
            model = WhisperModel(self.whisper_model, device=device_str)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

        # Transcribe video with word timestamps
        self.logger.info(f"Starting transcription of {self.Video_path}")
        try:
            segments, info = model.transcribe(self.Video_path, word_timestamps=True)
            segments = list(segments)
            self.logger.info(
                f"Transcription completed. Language: {info.language}, Segments: {len(segments)}"
            )
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise

        # Process transcription results into time-stamped words
        self.logger.info("Processing transcription results")
        time_stamped = []
        full_text = []
        word_count = 0

        for segment in segments:
            for word in segment.words:
                time_stamped.append([word.word, word.start, word.end])
                full_text.append(word.word)
                word_count += 1

        full_text = "".join(full_text)
        self.logger.info(f"Processed {word_count} words from {len(segments)} segments")
        self.logger.info(f"Full text length: {len(full_text)} characters")

        # Decompose Long Sentences

        # Tokenize the text into sentences
        self.logger.info("Tokenizing text into sentences")
        try:
            tokenized_sentences = sent_tokenize(full_text)
            sentences = []

            for i, sentence in enumerate(tokenized_sentences):
                sentences.append(sentence)

            self.logger.info(f"Successfully tokenized into {len(sentences)} sentences")
        except Exception as e:
            self.logger.error(f"Sentence tokenization failed: {str(e)}")
            raise

        # Map sentences to timestamps
        self.logger.info("Mapping sentences to timestamps")
        time_stamped_sentances = {}
        count_sentances = {}

        letter = 0
        for i in range(len(sentences)):
            tmp = []
            starts = []

            for j in range(len(sentences[i])):
                letter += 1
                tmp.append(sentences[i][j])

                f = 0
                for k in range(len(time_stamped)):
                    for m in range(len(time_stamped[k][0])):
                        f += 1

                        if f == letter:
                            starts.append(time_stamped[k][1])
                            starts.append(time_stamped[k][2])
            letter += 1

            if starts:  # Only add if we found timestamps
                time_stamped_sentances["".join(tmp)] = [min(starts), max(starts)]
                count_sentances[i + 1] = "".join(tmp)

        self.logger.info(
            f"Successfully mapped {len(time_stamped_sentances)} sentences to timestamps"
        )

        # Create final record structure
        record = []
        for sentence in time_stamped_sentances:
            record.append(
                [
                    sentence,
                    time_stamped_sentances[sentence][0],
                    time_stamped_sentances[sentence][1],
                ]
            )

        self.logger.info(f"Created record with {len(record)} time-stamped sentences")

        # Decompose Long Sentences

        """record = []
        for segment in transcript['segments']:
            print("#############################")
            sentance = []
            starts = []
            ends = []
            i = 1
            if len(segment['text'].split())>25:
                k = len(segment['text'].split())//4
            else:
                k = 25
            for word in segment['words']:
                if i % k != 0:
                    i += 1
                    sentance.append(word['word'])
                    starts.append(word['start'])
                    ends.append(word['end'])

                else:
                     i += 1
                     final_sentance = " ".join(sentance)
                     if starts and ends and final_sentance:
                         print(final_sentance+f'[{min(starts)} / {max(ends)}]')
                         record.append([final_sentance, min(starts), max(ends)])

                     sentance = []
                     starts = []
                     ends = []
            final_sentance = " ".join(sentance)
            if starts and ends and final_sentance:
                print(final_sentance+f'[{min(starts)} / {max(ends)}]')
                record.append([final_sentance, min(starts), max(ends)])
                sentance = []
                starts = []
                ends = []

        i = 1
        new_record = [record[0]]
        while i <len(record)-1:
            if len(new_record[-1][0].split()) +  len(record[i][0].split()) < 10:
                text = new_record[-1][0]+record[i][0]
                start = new_record[-1][1]
                end = record[i][2]
                del new_record[-1]
                new_record.append([text, start, end])
            else:
                new_record.append(record[i])
            i += 1"""

        new_record = record
        self.logger.info(f"Using {len(new_record)} sentences for processing")

        # Audio Emotions Analysis
        self.logger.info("Initializing emotion recognition system")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier = foreign_class(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    pymodule_file="custom_interface.py",
                    classname="CustomEncoderWav2vec2Classifier",
                    run_opts={"device": f"{device}"},
                )
            self.logger.info("Emotion recognition system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize emotion recognition: {str(e)}")
            raise

        emotion_dict = {
            "neu": "Neutral",
            "ang": "Angry",
            "hap": "Happy",
            "sad": "Sad",
            "None": None,
        }

        if not self.Context_translation:

            # Function to translate text
            def translate(sentence):
                if self.source_language == "tr":
                    model_name = f"Helsinki-NLP/opus-mt-trk-{self.target_language}"
                elif self.target_language == "tr":
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-trk"
                elif self.source_language == "zh-cn":
                    model_name = f"Helsinki-NLP/opus-mt-zh-{self.target_language}"
                elif self.target_language == "zh-cn":
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-zh"
                else:
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(device)

                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to(
                    device
                )
                translated = model.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)

        else:
            client = Groq(api_key=self.Context_translation)

            def translate(sentence, before_context, after_context, target_language):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                        Role: You are a professional translator who translates concisely in short sentence while preserving meaning.
                        Instruction:
                        Translate the given sentence into {target_language}


                        Sentence: {sentence}


                        Output format:
                        [[sentence translation: <your translation>]]
                        """,
                        }
                    ],
                    model="llama3-70b-8192",
                )
                # return chat_completion.choices[0].message.content
                # Regex pattern to extract the translation
                pattern = r"\[\[sentence translation: (.*?)\]\]"

                # Extracting the translation
                match = re.search(pattern, chat_completion.choices[0].message.content)

                try:
                    translation = match.group(1)
                    return translation
                except:
                    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

        records = []

        audio = AudioSegment.from_file(audio_file, format="mp4")
        for i in range(len(new_record)):
            final_sentance = new_record[i][0]
            if not self.Context_translation:
                translated = translate(sentence=final_sentance)

            else:
                before_context = (
                    new_record[i - 1][0] if i - 1 in range(len(new_record)) else ""
                )
                after_context = (
                    new_record[i + 1][0] if i + 1 in range(len(new_record)) else ""
                )
                translated = translate(
                    sentence=final_sentance,
                    before_context=before_context,
                    after_context=after_context,
                    target_language=self.target_language,
                )
            speaker = most_occured_speaker

            max_overlap = 0

            # Check overlap with each speaker's time range
            for key, value in speakers_rolls.items():
                speaker_start = int(key[0])
                speaker_end = int(key[1])

                # Calculate overlap
                overlap = get_overlap(
                    (new_record[i][1], new_record[i][2]), (speaker_start, speaker_end)
                )

                # Update speaker if this overlap is greater than previous ones
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = value

            start = int(new_record[i][1]) * 1000
            end = int(new_record[i][2]) * 1000

            try:
                audio[start:end].export("audio/emotions.wav", format="wav")
                out_prob, score, index, text_lab = classifier.classify_file(
                    "audio/emotions.wav"
                )
                os.remove("audio/emotions.wav")
            except:
                text_lab = ["None"]

            records.append(
                [
                    translated,
                    final_sentance,
                    new_record[i][1],
                    new_record[i][2],
                    speaker,
                    emotion_dict[text_lab[0]],
                ]
            )
            print(
                translated,
                final_sentance,
                new_record[i][1],
                new_record[i][2],
                speaker,
                emotion_dict[text_lab[0]],
            )

        os.environ["COQUI_TOS_AGREED"] = "1"
        if device == "cuda":
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        else:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        #!tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  --list_speaker_idxs

        os.system("rm -r audio_chunks")
        os.system("rm -r su_audio_chunks")
        os.system("mkdir audio_chunks")
        os.system("mkdir su_audio_chunks")

        natural_scilence = records[0][2]
        previous_silence_time = 0

        if natural_scilence >= 0.8:
            previous_silence_time = 0.8
            natural_scilence -= 0.8
        else:
            previous_silence_time = natural_scilence
            natural_scilence = 0

        combined = AudioSegment.silent(duration=natural_scilence * 1000)

        tip = 350

        def truncate_text(text, max_tokens=50):
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens]) + "..."

        for i in range(len(records)):
            print("previous_silence_time: ", previous_silence_time)
            tts.tts_to_file(
                text=truncate_text(records[i][0]),
                file_path=f"audio_chunks/{i}.wav",
                speaker_wav=f"speakers_audio/{records[i][4]}.wav",
                language=self.target_language,
                emotion=records[i][5],
                speed=2,
            )

            audio = AudioSegment.from_file(f"audio_chunks/{i}.wav")
            audio = audio[: len(audio) - tip]
            audio.export(f"audio_chunks/{i}.wav", format="wav")

            lt = len(audio) / 1000.0
            lo = max(records[i][3] - records[i][2], 0)
            theta = lo / lt

            input_file = f"audio_chunks/{i}.wav"
            output_file = f"su_audio_chunks/{i}.wav"

            if theta < 1 and theta > 0.44:
                print("############################")
                theta_prim = (lo + previous_silence_time) / lt
                command = f"ffmpeg -i {input_file} -filter:a 'atempo={1/theta_prim}' -vn {output_file}"
                process = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if process.returncode != 0:
                    sc = lo + previous_silence_time
                    silence = AudioSegment.silent(duration=(sc * 1000))
                    silence.export(output_file, format="wav")
            elif theta < 0.44:
                silence = AudioSegment.silent(
                    duration=((lo + previous_silence_time) * 1000)
                )
                silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(previous_silence_time * 1000))
                audio = silence + audio
                audio.export(output_file, format="wav")

            audio = AudioSegment.from_file(output_file)
            lt = len(audio) / 1000.0
            lo = records[i][3] - records[i][2] + previous_silence_time
            if i + 1 < len(records):
                natural_scilence = max(records[i + 1][2] - records[i][3], 0)
                if natural_scilence >= 0.8:
                    previous_silence_time = 0.8
                    natural_scilence -= 0.8
                else:
                    previous_silence_time = natural_scilence
                    natural_scilence = 0

                silence = AudioSegment.silent(
                    duration=((max(lo - lt, 0) + natural_scilence) * 1000)
                )
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(max(lo - lt, 0) * 1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")

            print("#######diff######: ", lo - lt)
            print("lo: ", lo)
            print("lt: ", lt)

        # Get all the audio files from the folder
        audio_files = [
            f
            for f in os.listdir("su_audio_chunks")
            if f.endswith((".mp3", ".wav", ".ogg"))
        ]

        # Sort files to concatenate them in order, if necessary
        audio_files.sort(
            key=lambda x: int(x.split(".")[0])
        )  # Modify sorting logic if needed (e.g., based on filenames)

        # Loop through and concatenate each audio file
        for audio_file in audio_files:
            file_path = os.path.join("su_audio_chunks", audio_file)
            audio_segment = AudioSegment.from_file(file_path)
            combined += audio_segment  # Append audio to the combined segment

        audio = AudioSegment.from_file(self.Video_path)
        total_length = len(audio) / 1000.0
        silence = AudioSegment.silent(
            duration=abs(total_length - records[-1][3]) * 1000
        )
        combined += silence
        # Export the combined audio to the output file
        combined.export("audio/output.wav", format="wav")

        # Initialize Spleeter with the 2stems model (vocals + accompaniment)
        separator = Separator()

        # Load a model
        separator.load_model(model_filename="2_HP-UVR.pth")
        output_file_paths = separator.separate(self.Video_path)[0]

        audio1 = AudioSegment.from_file("audio/output.wav")
        audio2 = AudioSegment.from_file(output_file_paths)
        combined_audio = audio1.overlay(audio2)

        # Export the combined audio file
        combined_audio.export("audio/combined_audio.wav", format="wav")

        # Video and Audio Overlay with error handling

        command = f"ffmpeg -err_detect ignore_err -fflags +igndts+ignidx -i '{self.Video_path}' -i audio/combined_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest -y output_video.mp4"
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"FFmpeg overlay failed, trying fallback method: {result.stderr}")
            # Fallback: try with different audio handling
            command = f"ffmpeg -i '{self.Video_path}' -i audio/combined_audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y output_video.mp4"
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        shutil.move(output_file_paths, "audio/")
        # os.system('pip install -r requirements.txt > /dev/null 2>&1')

        if self.Voice_denoising:

            """model, df_state, _ = init_df()
            audio, _ = load_audio("audio/combined_audio.wav", sr=df_state.sr())
            # Denoise the audio
            enhanced = enhance(model, df_state, audio)
            # Save for listening
            save_audio("audio/enhanced.wav", enhanced, df_state.sr())"""
            command = f"ffmpeg -err_detect ignore_err -fflags +igndts+ignidx -i '{self.Video_path}' -i audio/output.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest -y denoised_video.mp4"
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                print(
                    f"FFmpeg denoised overlay failed, trying fallback: {result.stderr}"
                )
                command = f"ffmpeg -i '{self.Video_path}' -i audio/output.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y denoised_video.mp4"
                subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        if self.LipSync and self.Voice_denoising:
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system(
                "cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../denoised_video.mp4' --audio '../audio/output.wav' --face_det_batch_size 1 --wav2lip_batch_size 1"
            )

        if self.LipSync and not self.Voice_denoising:
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system(
                "cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../output_video.mp4' --audio '../audio/combined_audio.wav' --face_det_batch_size 1 --wav2lip_batch_size 1"
            )

        if self.LipSync and self.Voice_denoising:
            source_path = "Wav2Lip/results/result_voice.mp4"
            destination_folder = "results"

            shutil.move(source_path, destination_folder)
            os.remove("output_video.mp4")
            shutil.move("denoised_video.mp4", destination_folder)

        elif self.LipSync and not self.Voice_denoising:
            source_path = "Wav2Lip/results/result_voice.mp4"
            destination_folder = "results"

            shutil.move(source_path, destination_folder)
            os.remove("output_video.mp4")
            os.remove("denoised_video.mp4")

        elif not self.LipSync and self.Voice_denoising:
            source_path = "denoised_video.mp4"
            destination_folder = "results"

            shutil.move(source_path, destination_folder)
            os.remove("output_video.mp4")
        else:
            source_path = "output_video.mp4"
            destination_folder = "results"

            shutil.move(source_path, destination_folder)


# Helper functions (these would need to be implemented or imported)
def get_speaker(time, speakers_rolls):
    """Get speaker for a given time"""
    # Implementation needed
    pass


def extract_frames(video_path, output_folder, speakers_rolls):
    """Extract frames from video"""
    # Implementation needed
    pass


def detect_and_crop_faces(image_path, face_cascade):
    """Detect and crop faces from image"""
    # Implementation needed
    pass


def extract_and_save_most_common_face(speaker_folder_path):
    """Extract most common face for a speaker"""
    # Implementation needed
    pass


def get_overlap(range1, range2):
    """Calculate overlap between two time ranges"""
    # Implementation needed
    pass
