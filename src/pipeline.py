import os
import pickle

import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook

from src.audiofile import AudioFile
from src.config import config
from src.util import StreamlitProgressHook

speaker_diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=config.hf_token
)

class SpeakerDiarizationPipeline:
    def __init__(self, audio_file: AudioFile):
        self.audio_file = audio_file
        self.device = self.get_device()
        self.pipeline = speaker_diarization_pipeline
        self.pipeline.to(self.device)

        self.metric_der = DiarizationErrorRate()
        self.metric_jer = JaccardErrorRate()
    
    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def diarize(self, streamlit_progress_hook: bool = False) -> Annotation:
        # Define the path for the saved diarization result
        diarization_path = self.audio_file.diarization_path

        # Check if the diarization result already exists
        if os.path.exists(diarization_path):
            print(f"Loading diarization from {diarization_path}")
            with open(diarization_path, 'rb') as f:
                diarization = pickle.load(f)
        else:
            print("Running diarization streamlit...")

            Hook = StreamlitProgressHook if streamlit_progress_hook else ProgressHook
            
            with Hook() as hook:
                diarization = self.pipeline(self.audio_file.audio_file_path, hook=hook)
            
            with open(diarization_path, 'wb') as f:
                pickle.dump(diarization, f)
            print(f"Diarization saved to {diarization_path}")

        return diarization

    def save_predicted_rttm(self, diarization: Annotation):
        # Use the audio file name for the predicted RTTM file
        with open(self.audio_file.pred_rttm_path, 'w') as f:
            diarization.write_rttm(f)
        print(f"Predicted RTTM saved to {self.audio_file.pred_rttm_path}")

    def calculate_der(self, diarization: Annotation):
        
        reference = self.audio_file.reference
        
        # Define window size for DER calculation (in seconds)
        window_size = 30.0  # 30-second windows
        step_size = 15.0  # Overlapping step size (15 seconds)

        # Get the total duration of the audio
        audio_duration = sum(segment.duration for segment in reference.get_timeline())

        # Initialize lists to store results
        cumulative_der = []
        cumulative_percentage = []

        # Calculate DER for cumulative segments
        processed_duration = 0.0
        start_time = 0.0

        while start_time < audio_duration:
            end_time = min(start_time + window_size, audio_duration)
            window = Segment(0.0, end_time)  # Cumulative segment from start to current window's end

            # Crop reference and diarization to the current cumulative window
            reference_window = reference.crop(window, mode="loose")
            diarization_window = diarization.crop(window, mode="loose")

            
            # Skip if the cumulative window is empty
            if not reference_window or not diarization_window:
                start_time += step_size
                continue

            # Compute DER for the cumulative window
            der = self.metric_der(reference_window, diarization_window)

            # Update processed duration and calculate percentage
            processed_duration = end_time
            percentage_processed = processed_duration / audio_duration * 100

            # Append to results
            cumulative_der.append(der * 100)
            cumulative_percentage.append(percentage_processed)

            # Move to the next window
            start_time += step_size

        # Plot DER as a function of percentage of audio processed
        return cumulative_percentage, cumulative_der

    def plot_der(self, cumulative_percentage, cumulative_der):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_percentage, cumulative_der, marker='o', label='Cumulative DER')
        ax.set_xlabel('Percentage of Audio Processed (%)')
        ax.set_ylabel('Diarization Error Rate (%)')
        ax.set_title('DER vs Percentage of Audio Processed')
        ax.grid(True)
        ax.legend()
        
        # Save the plot to the specified directory
        plot_path = os.path.join("outputs", "accuracy_plots", f"{self.audio_file.audio_filename}_der_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        return fig

    def calculate_jer(self, diarization: Annotation):
        reference = self.audio_file.reference
        
        # Define window size for JER calculation (in seconds)
        window_size = 30.0  # 30-second windows
        step_size = 15.0  # Overlapping step size (15 seconds)

        # Get the total duration of the audio
        audio_duration = sum(segment.duration for segment in reference.get_timeline())

        # Initialize lists to store results
        cumulative_jer = []
        cumulative_percentage = []

        # Calculate JER for cumulative segments
        processed_duration = 0.0
        start_time = 0.0

        while start_time < audio_duration:
            end_time = min(start_time + window_size, audio_duration)
            window = Segment(0.0, end_time)  # Cumulative segment from start to current window's end

            # Crop reference and diarization to the current cumulative window
            reference_window = reference.crop(window, mode="loose")
            diarization_window = diarization.crop(window, mode="loose")

            # Skip if the cumulative window is empty
            if not reference_window or not diarization_window:
                start_time += step_size
                continue

            # Compute JER for the cumulative window
            jer = self.metric_jer(reference_window, diarization_window)

            # Update processed duration and calculate percentage
            processed_duration = end_time
            percentage_processed = processed_duration / audio_duration * 100

            # Append to results
            cumulative_jer.append(jer * 100)
            cumulative_percentage.append(percentage_processed)

            # Move to the next window
            start_time += step_size

        # Return JER as a function of percentage of audio processed
        return cumulative_percentage, cumulative_jer

    def plot_jer(self, cumulative_percentage, cumulative_jer):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_percentage, cumulative_jer, marker='o', label='Cumulative JER')
        ax.set_xlabel('Percentage of Audio Processed (%)')
        ax.set_ylabel('Jaccard Error Rate (%)')
        ax.set_title('JER vs Percentage of Audio Processed')
        ax.grid(True)
        ax.legend()
        
        # Save the plot to the specified directory
        plot_path = os.path.join("outputs", "accuracy_plots", f"{self.audio_file.audio_filename}_jer_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        return fig




if __name__ == "__main__": 
    # Paths
    audio_file = "dataset/audio/11.mp3"
    transcript_file = "dataset/transcript/11.json"
    output_dir = "outputs"

    # Initialize DataLoader and generate RTTM
    audio_file_instance = AudioFile(audio_file, transcript_file, output_dir)
    audio_file_instance.generate_true_rttm()

    # Initialize and execute the pipeline
    pipeline = SpeakerDiarizationPipeline(audio_file_instance)
    diarization = pipeline.diarize()
    pipeline.save_predicted_rttm(diarization)
    cumulative_percentage, cumulative_der = pipeline.calculate_der(diarization)
    fig = pipeline.plot_der(cumulative_percentage, cumulative_der)

    # Show the plot when running this script directly
    plt.show()
