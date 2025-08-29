import os
import gradio as gr
from video_dubbing import VideoDubbing

# Language mapping
language_mapping = {
    'English': 'en', 
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de', 
    'Italian': 'it', 
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Hindi': 'hi', 
    'Hungarian': 'hu'
}

def process_video(video_file, video_url, source_language, target_language, use_wav2lip, whisper_model, bg_sound):
    try:
        os.system("rm video_path.mp4")
        video_path = None
        
        # Handle video URL (YouTube)
        if video_url and "youtube.com" in video_url:
            os.system(f"yt-dlp -f best -o 'video_path.mp4' --recode-video mp4 {video_url}")
            video_path = "video_path.mp4"
        # Handle uploaded video file
        elif video_file:
            video_path = video_file
        else:
            return None, "Error: Please provide either a video file or YouTube URL"
        
        vidubb = VideoDubbing(video_path, language_mapping[source_language], language_mapping[target_language], use_wav2lip, not bg_sound, whisper_model, "", os.getenv('HF_TOKEN'))
        if  use_wav2lip and not bg_sound:
            source_path = 'results/result_voice.mp4'
                

        elif use_wav2lip and bg_sound:
            source_path = 'results/result_voice.mp4'

        
        elif not use_wav2lip and not bg_sound:
            source_path = 'results/denoised_video.mp4'

        else:
            source_path = 'results/output_video.mp4'
        
        return source_path, "No Error"

    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return None, f"Error: {str(e)}"

def create_gradio_interface():
    """Create and return the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ViDubb")
        gr.Markdown("This tool uses AI to dub videos into different languages!")
        
        with gr.Row():
            with gr.Column(scale=2):
                    video_file = gr.Video(label="Upload Video (Optional)",height=500, width=500)
                    video_url = gr.Textbox(label="YouTube URL (Optional)", placeholder="Enter YouTube URL")
                    source_language = gr.Dropdown(
                        choices=list(language_mapping.keys()),  # You can use `language_mapping.keys()` here
                        label="Source Language for Dubbing",
                        value="English"
                    )
                    target_language = gr.Dropdown(
                        choices=list(language_mapping.keys()),  # You can use `language_mapping.keys()` here
                        label="Target Language for Dubbing",
                        value="French"
                    )
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        label="Whisper Model",
                        value="medium"
                    )
                    use_wav2lip = gr.Checkbox(
                        label="Use Wav2Lip for lip sync",
                        value=False,
                        info="Enable this if the video has close-up faces. May not work for all videos."
                    )
                    
                    bg_sound = gr.Checkbox(
                        label="Keep Background Sound",
                        value=False,
                        info="Keep background sound of the original video, may introduce noise."
                    )
                    submit_button = gr.Button("Process Video", variant="primary")
            
            with gr.Column(scale=2):
                output_video = gr.Video(label="Processed Video",height=500, width=500)
                error_message = gr.Textbox(label="Status/Error Message")

        submit_button.click(
            process_video, 
            inputs=[video_file, video_url, source_language, target_language, use_wav2lip, whisper_model, bg_sound], 
            outputs=[output_video, error_message]
        )

    return demo

def launch_interface():
    """Launch the Gradio interface"""
    print("Launching Gradio interface...")
    demo = create_gradio_interface()
    demo.queue()
    demo.launch(share=True)