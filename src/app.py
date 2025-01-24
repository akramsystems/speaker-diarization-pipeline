import os

import streamlit as st


from src.audiofile import AudioFile
from src.pipeline import SpeakerDiarizationPipeline

# Streamlit app title
st.set_page_config(
    page_title="Who Said What?",
    page_icon="🤷"
)


st.title(" 🎙️ Speaker Diarization ")

# File uploader for audio file
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
# File uploader for transcript file
transcript_file = st.file_uploader("Upload Transcript File", type=["json"])

# Directory to save uploaded files
input_dir = "dataset"
output_dir = "outputs"


if audio_file and transcript_file:
    # Save uploaded files
    audio_file_path = os.path.join(os.path.join(input_dir, "audio"), audio_file.name)
    transcript_file_path = os.path.join(os.path.join(input_dir, "transcript"), transcript_file.name)

    with open(audio_file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    with open(transcript_file_path, "wb") as f:
        f.write(transcript_file.getbuffer())

    # Initialize AudioFile instance
    audio_file_instance = AudioFile(audio_file_path, transcript_file_path, output_dir)
    audio_file_instance.generate_true_rttm()

    # Initialize and execute the pipeline with progress bar
    pipeline = SpeakerDiarizationPipeline(audio_file_instance)
    diarization = pipeline.diarize(streamlit_progress_hook=True)
    pipeline.save_predicted_rttm(diarization)
    cumulative_percentage, cumulative_der = pipeline.calculate_der(diarization)

    # Plot DER
    st.subheader("Diarization Error Rate (DER) Plot")
    fig = pipeline.plot_der(cumulative_percentage, cumulative_der)
    st.pyplot(fig)