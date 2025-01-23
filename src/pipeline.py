import os
import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate, GreedyMapper
from pyannote.core import Annotation, Segment
from pydub import AudioSegment
import pickle

from src.dataloader import DataLoader
from src.config import config


class SpeakerDiarizationPipeline:
    def __init__(self, audio_file, true_rttm_path, output_dir):
        # Convert mp3 to wav
        if audio_file.endswith('.mp3'):
            audio_file_wav = audio_file.replace('.mp3', '.wav')
            if not os.path.exists(audio_file_wav):
                audio = AudioSegment.from_mp3(audio_file)
                audio.export(audio_file_wav, format='wav')
            audio_file = audio_file_wav
        
        self.audio_file = audio_file
        self.audio_filename = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.diarization_dir = os.path.join(self.output_dir, "diarization_results")
        os.makedirs(self.diarization_dir, exist_ok=True)
        self.true_rttm_path = true_rttm_path
        self.output_dir = output_dir
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.hf_token)
        self.num_speakers = self.get_number_of_speakers()
    
    def get_number_of_speakers(self) -> int:
        speakers = set()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                speaker = parts[7]
                speakers.add(speaker)
        return len(speakers)

    def diarize(self) -> Annotation:
        # Define the path for the saved diarization result
        diarization_path = os.path.join(self.diarization_dir, f"{self.audio_filename}.pkl")

        # Check if the diarization result already exists
        if os.path.exists(diarization_path):
            print(f"Loading diarization from {diarization_path}")
            with open(diarization_path, 'rb') as f:
                diarization = pickle.load(f)
        else:
            print("Running diarization...")
            self.pipeline.to(torch.device("mps"))
            diarization = self.pipeline(self.audio_file, num_speakers=self.num_speakers)
            
            # Save the diarization result
            with open(diarization_path, 'wb') as f:
                pickle.dump(diarization, f)
            print(f"Diarization saved to {diarization_path}")

        return diarization

    def save_predicted_rttm(self, diarization: Annotation):
        pred_rttm_dir = os.path.join(self.output_dir, "pred_rttm")
        os.makedirs(pred_rttm_dir, exist_ok=True)
        
        # Use the audio file name for the predicted RTTM file
        pred_rttm_path = os.path.join(pred_rttm_dir, f"{self.audio_filename}.rttm")
        
        with open(pred_rttm_path, 'w') as f:
            diarization.write_rttm(f)
        print(f"Predicted RTTM saved to {pred_rttm_path}")

    def calculate_der(self, diarization: Annotation):
        # Load ground truth RTTM
        reference = Annotation()
        with open(self.true_rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
                reference[Segment(start, start + duration)] = speaker

        # Define window size for DER calculation (in seconds)
        window_size = 30.0  # 30-second windows
        step_size = 15.0  # Overlapping step size (15 seconds)

        # Get the total duration of the audio
        audio_duration = sum(segment.duration for segment in reference.get_timeline())

        # Instantiate the DiarizationErrorRate metric
        metric = DiarizationErrorRate()

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
            der = metric(reference_window, diarization_window)

            # Update processed duration and calculate percentage
            processed_duration = end_time
            percentage_processed = processed_duration / audio_duration * 100

            # Append to results
            cumulative_der.append(der * 100)
            cumulative_percentage.append(percentage_processed)

            # Move to the next window
            start_time += step_size

        # Plot DER as a function of percentage of audio processed
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_percentage, cumulative_der, marker='o', label='Cumulative DER')
        plt.xlabel('Percentage of Audio Processed (%)')
        plt.ylabel('Diarization Error Rate (%)')
        plt.title('DER vs Percentage of Audio Processed')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Print the overall DER for reference
        overall_der = metric(reference, diarization)
        print(f"Overall Diarization Error Rate (DER): {overall_der * 100:.2f}%")
        return overall_der


if __name__ == "__main__": 
    # Paths
    audio_file = "dataset/audio/11.mp3"
    transcript_file = "dataset/transcript/11.json"
    output_dir = "outputs"

    # Initialize DataLoader and generate RTTM
    data_loader = DataLoader(audio_file, transcript_file, output_dir)
    data_loader.generate_true_rttm()
    true_rttm_path = data_loader.get_true_rttm_path()

    # Initialize and execute the pipeline
    pipeline = SpeakerDiarizationPipeline(audio_file, true_rttm_path, output_dir)
    diarization = pipeline.diarize()
    pipeline.save_predicted_rttm(diarization)
    pipeline.calculate_der(diarization)
